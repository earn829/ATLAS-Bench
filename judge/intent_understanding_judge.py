"""Intent-understanding judge (DeepSeek-Pro via the judge gateway).

Scores ONE dimension: whether the agent, at every step, is still solving the CORRECT
problem (task type, target subject/category, time context, explicit place-anchor vs
"near me"), identifies all explicit hard constraints + main soft preferences with
correct priority, decomposes compound needs into subtasks, ties each tool call to an
open subtask and updates state from observations, and does NOT drift (forget a
constraint, swap the anchor, upgrade soft->hard, or silently relax a hard constraint).

Same anti-inflation design as judge/tool_flexibility_judge.py: integer 1-5, default
3, and validate() enforces caps keyed on concrete cited failures (any missed hard
constraint or intent-drift event forces the score to 1-2; >=4 requires none; 5 also
requires zero soft misses). Final-answer correctness is out of scope.
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
from paired_acceptability_judge import parse_json  # noqa: E402
from tool_flexibility_judge import compact_trajectory  # noqa: E402

SYSTEM = '''你是严格的「逐步意图理解(intent_understanding)」评审员。输入包含：用户 query、上下文(time/loc)、该 query 的人工约束清单(constraints，每条带 cid/text/h=刚性|柔性，作为「应识别的意图」参照)、一个匿名 agent 的完整工具调用轨迹(逐轮：工具名、参数 arguments、该轮 rationale、结果数量/是否为空)、以及最终答案概要(summary、推荐数、no_exact_match、各推荐名称与其 constraint_gaps)。

【只评一个维度：逐步意图理解】即 agent 在每一步是否仍在解决「正确的问题」，而不只是动作表面相关。重点核查：
1. 是否正确理解任务类型、目标主体/品类、时间语境；尤其地点锚点——query 含明确地标(「XX附近/这边/一带」、具体商圈/街道/地标)时，必须把检索锚定到该地标(把地标名写进检索 query 文本、或用 district/business_area 过滤该地标、或先定位地标坐标再周边搜，任一方式即可)，不得自始至终只用用户原始坐标而完全不引用地标；query 为「我附近/我身边/离我近」这类以用户自身为锚(无明确地标)时，直接用用户坐标检索是正确的、不算失误；
2. 是否识别了全部显式刚性约束(h=刚性)与主要柔性偏好(h=柔性)，并保持正确优先级(刚性优先于柔性)；
3. 是否把复合需求拆成必要子任务(候选召回、属性核验、UGC/笔记核验、距离/路线、比较、动态状态等)；
4. 每次工具调用是否对应一个尚未解决的子任务，看到观测后是否更新了对任务状态的判断；
5. 是否出现意图漂移：遗忘约束、偷换地点锚点(把明确地标替换为用户位置)、把软偏好升级成硬条件、或静默放松刚性约束。

【不要评】最终答案是否真的满足约束(那是可接受性维度)、事实是否准确、检索召回质量、工具灵活度、并行度。本维度只看「是否始终正确理解并 pursuit 用户意图」。

【打分 1–5(整数)，严格锚定，默认 3 分；必须先列证据再打分】
定义「实质性意图失误(material)」= 漏掉或放松了任一刚性约束(记入 hard_missed)，或出现任一意图漂移事件(记入 intent_errors：偷换锚点 / 软升硬 / 静默放松刚性 / 遗忘约束 / 品类·时间·主体理解错误)。
- 5(极少给)：零 material 失误，且 soft_missed 为空；全程紧扣意图，子任务拆解干净，每次调用都对应未解决子任务并据观测更新状态。
- 4：零 material 失误，但有 ≤1 处柔性偏好遗漏或轻微子任务瑕疵。
- 3(最常见)：零 material 失误，但意图追踪平淡——多处柔性遗漏、拆解不完整、或有调用未明确对应子任务。
- 2：恰好 1 处 material 失误。
- 1：≥2 处 material 失误，或从头到尾根本性误解任务(品类/主体/时间/锚点全程错)。

【地点锚点 anchor 字段(必填，四选一)】判定 agent 是否把检索锚定到 query 指名的地标。锚定地标的合法方式有以下几种，用了任意一种即算正确锚定：(i) 在 search_pois/search_notes 的 query 文本里带上地标名(如 query="姚家园 西餐厅")；(ii) 用 district/business_area 过滤为该地标(如 district="东坝")；(iii) 先定位地标坐标、再以该坐标做 search_nearby。
- "ok"：用上述任一方式锚定了地标(即使另外并行或补充做了一次以用户坐标为中心的 nearby 宽召回，仍算 ok)；或 query 本是「我附近/我身边/离我近」这类(无明确地标)而直接用用户坐标。
- "na"：query 不含明确地标锚点。
- "self_corrected"：起初完全没引用地标(只按用户坐标搜)，随后主动改用上述任一方式锚定地标。轻微疏漏——不计入 intent_errors，分数最高只能给 3。
- "wrong_uncorrected"：全程从未用任何方式引用该地标(query 文本、district/business_area 都没出现地标名)，自始至终只用用户坐标检索(即「一直用用户位置搜」)。实质失误——必须计入 intent_errors，分数只能 1–2。
判定 anchor 时务必逐轮检查 search_pois/search_notes 的 query 文本与 district/business_area 参数里有没有出现地标名，不要只盯着 search_nearby 的 loc 是不是用户坐标。

【反通胀硬性规则，违反将判为无效输出并要求重打】
1. 只要 hard_missed 或 intent_errors 非空(即存在 material 失误)，分数只能是 1 或 2。
2. 分数 ≥4 必须 hard_missed 与 intent_errors 均为空、且 anchor 为 ok 或 na；分数=5 还须 soft_missed 为空。
3. 分数 ≤2 必须给出 ≥1 条具体的 hard_missed 或 intent_errors(带 cid 或轮次引用)。
4. anchor=wrong_uncorrected 必须在 intent_errors 中列出该锚点偷换，且分数只能 1–2。
5. anchor=self_corrected 分数 ≤3，且不得把这条已纠正的锚点问题计入 intent_errors(避免双重惩罚)。
6. 「调用多/答案长」≠「意图理解好」；表面相关但偏离真实意图的动作要扣分。
7. 不得因最终答案好坏加减分；只评意图理解过程。hard_missed / soft_missed / intent_errors 必须是具体、可核对、带 cid 或轮次引用的短语；确实没有则给空数组 []。

只输出合法 JSON，不要多余文字：
{"task_id":"...","intent_score":3,"anchor":"ok","hard_missed":["c2:人均≤25被静默放宽到40"],"soft_missed":["c4:安静偏好未识别"],"intent_errors":["第3轮:把锚点太阳宫偷换为国贸"],"reason":"一句话总评"}'''


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
            {"cid": c.get("cid"), "text": c.get("text"), "h": c.get("h")}
            for c in (rec.get("constraints") or [])
        ]
    return out


def build_payload(record: Dict[str, Any], task_id: str, constraints: list) -> Dict[str, Any]:
    answer = record.get("answer") or {}
    inp = record.get("input") or {}
    recs = []
    for r in answer.get("recommendations") or []:
        recs.append({"name": r.get("name"), "constraint_gaps": r.get("constraint_gaps") or []})
    return {
        "task_id": task_id,
        "query": inp.get("query"),
        "context": {k: v for k, v in inp.items() if k != "query"},
        "constraints": constraints,
        "trajectory": compact_trajectory(record),
        "final": {
            "summary": answer.get("summary"),
            "n_recommendations": len(answer.get("recommendations") or []),
            "no_exact_match": answer.get("no_exact_match"),
            "recommendations": recs,
        },
    }


def validate(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("not an object")
    score = value.get("intent_score")
    if not isinstance(score, int) or isinstance(score, bool) or not 1 <= score <= 5:
        raise ValueError("intent_score must be int 1..5")
    def strlist(key):
        v = value.get(key)
        if not isinstance(v, list) or not all(isinstance(x, str) and x.strip() for x in v):
            raise ValueError(f"{key} must be list of nonempty str")
        return [x.strip() for x in v]
    hard = strlist("hard_missed")
    soft = strlist("soft_missed")
    errs = strlist("intent_errors")
    reason = value.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason must be nonempty str")
    anchor = value.get("anchor")
    if anchor not in ("ok", "self_corrected", "wrong_uncorrected", "na"):
        raise ValueError("anchor must be ok/self_corrected/wrong_uncorrected/na")
    n_mat = len(hard) + len(errs)
    if n_mat >= 1 and not (1 <= score <= 2):
        raise ValueError("material intent failure (hard_missed/intent_errors) forces score 1-2")
    if score >= 4 and n_mat != 0:
        raise ValueError("score>=4 requires zero hard_missed and zero intent_errors")
    if score == 5 and len(soft) != 0:
        raise ValueError("score=5 requires zero soft_missed")
    if score <= 2 and n_mat < 1:
        raise ValueError("score<=2 requires >=1 hard_missed or intent_error")
    if anchor == "wrong_uncorrected" and not (score <= 2 and len(errs) >= 1):
        raise ValueError("anchor=wrong_uncorrected requires score<=2 and the swap cited in intent_errors")
    if anchor == "self_corrected" and score > 3:
        raise ValueError("anchor=self_corrected caps score at 3")
    if score >= 4 and anchor not in ("ok", "na"):
        raise ValueError("score>=4 requires anchor ok or na")
    return {
        "task_id": value.get("task_id"),
        "intent_score": score,
        "anchor": anchor,
        "hard_missed": hard,
        "soft_missed": soft,
        "intent_errors": errs,
        "reason": reason.strip(),
    }


def call_judge(args: argparse.Namespace, api_key: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
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
                    counts[result["judgment"]["intent_score"]] += 1
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
