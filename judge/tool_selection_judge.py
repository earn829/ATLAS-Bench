"""Tool-selection judge (DeepSeek-Pro via the judge gateway).

Scores ONE dimension: whether the agent picks capability-matched tools per subtask and
sets query terms / poiid / coordinates / radius / time / category / price filters
correctly, with sound dependency order, independent same-round parallel calls, and
cause-driven adaptation after empty results. constraints[].tool is only a suggestion;
functionally-equivalent paths get full credit.

Two material failures cap the score at 1-2:
  A. a core-tool / key-parameter error that breaks the main retrieval (tool_errors);
  B. an applicable verification tool exists but the agent never ran a validity check
     for its preferred/any recommendation (unverified=true).
Critical exemption C: if the agent DID verify with the right tool but the result was
negative/empty/insufficient, that is NOT "unverified" (verified_but_negative=true,
unverified must be false) and must NOT trigger the cap.

Same anti-inflation design as the other judges: 1-5, default 3, caps enforced in validate().
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Dict

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "scripts"))
sys.path.insert(0, str(_HERE))

from finalize_run import dedupe_records, read_jsonl  # noqa: E402
from paired_acceptability_judge import compact_answer, parse_json  # noqa: E402
from tool_flexibility_judge import compact_trajectory  # noqa: E402

SYSTEM = '''你是严格的「工具选择(tool_selection)」评审员。输入：用户 query、上下文(time/loc)、人工约束清单(constraints，每条含 cid/text/h=刚性|柔性/evidence_type/tool[建议工具，可能缺省])、匿名 agent 的完整工具调用轨迹(逐轮：工具名、参数 arguments、rationale、结果数量/是否为空、少量样例)、recommendation_evidence(每条推荐的 observed_facts=该 POI 在轨迹中被任意工具返回过的全部字段、cited_evidence=被引用证据)、最终答案概要(summary、推荐数、no_exact_match、各推荐 poiid/name/引用证据/constraint_gaps)。

【只评一个维度：工具选择与参数正确性】即 agent 是否为各子任务选择了能力匹配的工具，并正确设置查询词、poiid、坐标、半径、时间、类别、价格等过滤参数。核查：
1. 工具与子任务匹配：候选召回用 POI/周边搜索；体验口碑用笔记搜索；结构化属性(价格/评分/营业时间/距离)用 POI 详情或结构化字段；路线/通勤用路线工具。
2. 参数正确：query 关键词、category、district/business_area、loc 坐标、distance_m 半径、time、价格/评分过滤等设置得当，不因参数过窄/错位/拼写错误而漏召。
3. 依赖顺序合理：先拿到候选或坐标，再查详情/笔记/路线，不能倒置。
4. 同轮并行调用必须彼此独立(不依赖同轮另一调用的输出)；必要的交叉核验不算冗余。
5. constraints[].tool 仅为建议：功能等价的工具路径同样可得满分，不因未用建议工具而扣分。
6. 空结果后应依据原因改关键词/范围/工具，不能无变化地重复同一调用。

【两类实质性失误(material，触发分数上限 1–2)】
A. 核心工具或关键参数错误导致主检索失效：如主品类/坐标/半径/关键词设错，使本应召回的候选检索不到(记入 tool_errors，带轮次/参数)。
B. 有适用工具却从未核验：对某条推荐的某个可判定刚性约束，没有任何工具返回过判定它所需的字段/证据(查 recommendation_evidence.observed_facts 与轨迹结果)，agent 却仍下了推荐结论(unverified=true)。注意：核验不必是单独的 get_poi_detail——若 search_pois/search_nearby 的结果已带回该字段(如 category/category_path/open_time/average_cost/rating/distance_m)，即视为已核验，不得判 unverified；只有当所需字段在全部工具返回中都缺失时才算未核验。

【关键豁免，不得据此扣分或触发上限】
C. 若 agent 已用正确工具完成核验、但结果为否定/空/证据不足——这不属于「未核验」(verified_but_negative=true 且 unverified 必须=false)，不得触发工具分上限，按其余表现正常打分。
D. 无推荐(空答案/no_exact_match)时，B 类「未核验推荐」不适用(unverified=false)。
E. 未用 constraints[].tool 建议工具但走了功能等价路径：不扣分。

【打分 1–5(整数)，严格锚定，默认 3；先列证据再打分】
- 5(极少给)：每个子任务都用能力匹配的工具+正确参数；依赖顺序合理；并行调用相互独立；空结果后有据地调整；对推荐做了核验(或核验合理返回否定/空，见C)；零 material 失误、零冗余、零参数瑕疵。
- 4：工具/参数基本正确，仅有 ≥1 处未破坏主检索、也未跳过必要核验的轻微参数瑕疵或冗余。
- 3(最常见)：工具大体合适但有明显不足——参数次优、有冗余、依赖顺序欠佳、或核验做得弱；主检索仍能运作。
- 2：恰好 1 处 material 失误(A 或 B)。
- 1：≥2 处 material 失误，或全程工具选择根本错误致主检索自始不可用。

【反通胀硬性规则，违反将判为无效输出并要求重打】
1. tool_errors 非空 或 unverified=true(存在 material 失误)→ 分数只能 1 或 2。
2. 分数 ≥4 必须 tool_errors 为空且 unverified=false；分数=5 还须 param_issues 与 redundant_calls 均为空。
3. 分数 ≤2 必须存在 material 失误(tool_errors 非空或 unverified=true)，并给出具体轮次/参数引用。
4. verified_but_negative=true 时 unverified 必须=false，且不得仅因「核验返回否定/空」压低分数(见C)。
5. 「调用多/工具杂」≠「工具选得好」；无变化重复、依赖错序、参数错配都要扣分。
6. 不得因最终答案是否满足约束而加减分(那是可接受性维度)；本维度只看工具选择与参数。tool_errors/param_issues/redundant_calls 必须是具体、可核对、带轮次或参数引用的短语；没有则给空数组 []。

只输出合法 JSON，不要多余文字：
{"task_id":"...","tool_score":3,"tool_errors":["第2轮:category=咖啡误设致主检索0结果"],"param_issues":["distance_m=300过小"],"redundant_calls":["第4轮重复第3轮相同search_pois"],"unverified":false,"verified_but_negative":true,"reason":"一句话总评"}'''


def load_run(path: Path) -> Dict[str, Dict[str, Any]]:
    records, _, parse_errors = dedupe_records(read_jsonl(path))
    if parse_errors:
        raise ValueError("%s 存在无法解析的行：%s" % (path, dict(parse_errors)))
    return {record["task_id"]: record for record in records}


def load_constraints(path: str) -> Dict[str, list]:
    raw = json.load(open(path, encoding="utf-8"))
    out: Dict[str, list] = {}
    for item in raw:
        rec = item["record"]
        out[rec["id"]] = [
            {"cid": c.get("cid"), "text": c.get("text"), "h": c.get("h"),
             "evidence_type": c.get("evidence_type"), "tool": c.get("tool")}
            for c in (rec.get("constraints") or [])
        ]
    return out


def build_payload(record: Dict[str, Any], task_id: str, constraints: list) -> Dict[str, Any]:
    answer = record.get("answer") or {}
    inp = record.get("input") or {}
    recs = [
        {"poiid": r.get("poiid"), "name": r.get("name"),
         "evidence": r.get("evidence") or [], "constraint_gaps": r.get("constraint_gaps") or []}
        for r in answer.get("recommendations") or []
    ]
    return {
        "task_id": task_id,
        "query": inp.get("query"),
        "context": {k: v for k, v in inp.items() if k != "query"},
        "constraints": constraints,
        "trajectory": compact_trajectory(record),
        "recommendation_evidence": compact_answer(record),
        "final": {
            "summary": answer.get("summary"),
            "n_recommendations": len(recs),
            "no_exact_match": answer.get("no_exact_match"),
            "recommendations": recs,
        },
    }


def validate(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("not an object")
    score = value.get("tool_score")
    if not isinstance(score, int) or isinstance(score, bool) or not 1 <= score <= 5:
        raise ValueError("tool_score must be int 1..5")
    def strlist(key):
        v = value.get(key)
        if not isinstance(v, list) or not all(isinstance(x, str) and x.strip() for x in v):
            raise ValueError(f"{key} must be list of nonempty str")
        return [x.strip() for x in v]
    tool_errors = strlist("tool_errors")
    param_issues = strlist("param_issues")
    redundant = strlist("redundant_calls")
    unverified = value.get("unverified")
    vbn = value.get("verified_but_negative")
    if not isinstance(unverified, bool):
        raise ValueError("unverified must be bool")
    if not isinstance(vbn, bool):
        raise ValueError("verified_but_negative must be bool")
    reason = value.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason must be nonempty str")
    if vbn and unverified:
        raise ValueError("verified_but_negative and unverified are mutually exclusive (rule C)")
    material = (len(tool_errors) >= 1) or unverified
    if material and not (1 <= score <= 2):
        raise ValueError("material tool failure (tool_errors/unverified) forces score 1-2")
    if score >= 4 and material:
        raise ValueError("score>=4 requires no tool_errors and unverified=false")
    if score == 5 and (param_issues or redundant):
        raise ValueError("score=5 requires zero param_issues and zero redundant_calls")
    if score <= 2 and not material:
        raise ValueError("score<=2 requires a material failure (tool_errors or unverified)")
    return {
        "task_id": value.get("task_id"), "tool_score": score,
        "tool_errors": tool_errors, "param_issues": param_issues, "redundant_calls": redundant,
        "unverified": unverified, "verified_but_negative": vbn, "reason": reason.strip(),
    }


def call_judge(args: argparse.Namespace, api_key: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
    last: Exception | None = None
    for attempt in range(args.judge_retries):
        body = {"model": args.judge_model, "messages": messages,
                "temperature": args.judge_temperature, "max_tokens": args.judge_max_tokens}
        request = urllib.request.Request(
            args.judge_url, data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + api_key}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=args.judge_timeout) as response:
                raw = json.loads(response.read().decode("utf-8"))
            judgment = validate(parse_json(raw["choices"][0]["message"]["content"]))
            judgment["task_id"] = payload["task_id"]
            return {"status": "completed", "judgment": judgment, "usage": raw.get("usage") or {}}
        except urllib.error.HTTPError as exc:
            last = exc
            if attempt + 1 < args.judge_retries:
                time.sleep(min(2 ** attempt, 8))
        except Exception as exc:
            last = exc
            if attempt + 1 < args.judge_retries:
                messages.append({"role": "user", "content": f"上一输出无效：{exc}。只返回符合schema的JSON。"})
    return {"status": "failed", "error": {"type": type(last).__name__, "message": str(last)}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tasks", default=str(Path(__file__).resolve().parents[1] / "query" / "subset_1200.json"))
    parser.add_argument("--judge-url", default=os.environ.get("JUDGE_URL") or os.environ.get("ATLAS_JUDGE_URL"),
                        help="Judge endpoint; defaults to $JUDGE_URL / $ATLAS_JUDGE_URL. Required unless --dry-run.")
    parser.add_argument("--judge-model", default="deepseek-v4-pro")
    parser.add_argument("--judge-api-key-env", default="ATLAS_JUDGE_API_KEY")
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--judge-max-tokens", type=int, default=2048)
    parser.add_argument("--judge-timeout", type=int, default=900)
    parser.add_argument("--judge-retries", type=int, default=5)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run = load_run(Path(args.run))
    cons = load_constraints(args.tasks)
    order = list(run.keys())
    if args.limit is not None:
        order = order[: args.limit]
    if args.dry_run:
        for task_id in order:
            print(json.dumps(build_payload(run[task_id], task_id, cons.get(task_id, [])), ensure_ascii=False, indent=2))
        return

    if not args.judge_url:
        raise SystemExit("--judge-url or JUDGE_URL is required (except with --dry-run)")
    api_key = os.environ.get(args.judge_api_key_env, "")
    if not api_key:
        raise SystemExit(f"{args.judge_api_key_env} is not set")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if output.exists() and not args.fresh:
        for line in read_jsonl(output):
            if line.strip():
                r = json.loads(line)
                if r.get("status") == "completed":
                    done.add(r["task_id"])
    pending = [t for t in order if t not in done]
    print(json.dumps({"selected": len(order), "pending": len(pending), "model": args.judge_model}, ensure_ascii=False), flush=True)

    def work(task_id: str) -> Dict[str, Any]:
        payload = build_payload(run[task_id], task_id, cons.get(task_id, []))
        result = call_judge(args, api_key, payload)
        result.update({"task_id": task_id, "label": args.label})
        return result

    lock = threading.Lock()
    counts: Counter = Counter()
    with output.open("w" if args.fresh else "a", encoding="utf-8") as handle:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(work, t): t for t in pending}
            for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
                result = future.result()
                with lock:
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    handle.flush()
                if result["status"] == "completed":
                    counts[result["judgment"]["tool_score"]] += 1
                else:
                    counts["failed"] += 1
                if index % 100 == 0:
                    print(f"{index}/{len(pending)}", flush=True)
    judged = sum(counts[s] for s in range(1, 6))
    mean = round(sum(s * counts[s] for s in range(1, 6)) / judged, 4) if judged else None
    print(json.dumps({"label": args.label, "judged": judged, "failed": counts["failed"],
                      "score_dist": {str(s): counts[s] for s in range(1, 6)}, "mean_score": mean}, ensure_ascii=False))


if __name__ == "__main__":
    main()
