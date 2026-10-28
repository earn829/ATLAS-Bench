"""Evidence-sufficiency judge (DeepSeek-Pro via the judge gateway), 0-5 scale.

Evaluates whether the evidence ACTUALLY OBTAINED by trajectory end can verify each user
need / annotated constraint -- not the final wording. Per constraint the judge assigns
status in {sufficient, partial, missing, conflicting, expired}; evidence must map to the
correct POI/branch and be traceable and time-valid (notes created after the query time are
expired; static open_time cannot prove real-time availability; straight-line distance cannot
prove route time). Comparison/superlative tasks need a comparable candidate set (verifying a
single candidate then claiming "best" is a coverage failure). A limited top-k or one empty
result only proves "this search found nothing", not "globally absent".

Anti-inflation caps enforced in validate() using hard_deficient = # hard constraints whose
status != sufficient:
  hard_deficient>=2 -> score 0 ; ==1 -> score 1 ; ==0 -> score>=2
  comparison_single_candidate -> score<=2 ; fabricated_or_wrong_object -> score<=1
  score>=4 -> all hard AND soft sufficient, no single-candidate comparison, no fabrication.
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
from paired_acceptability_judge import evidence_catalog, parse_json  # noqa: E402

STATUSES = {"sufficient", "partial", "missing", "conflicting", "expired"}

SYSTEM = '''你是严格的「证据充分性(evidence_sufficiency)」评审员。评估**轨迹结束时 agent 实际获得的证据**能否逐条验证用户需求与人工约束——**不评价最终文案/推荐措辞**，只看证据本身够不够、对不对、新不新。

输入：query、上下文(含查询 time、loc)、人工约束清单(constraints：cid/text/h=刚性|柔性/evidence_type)、agent 全程观测到的证据池 all_observed_evidence(每条含 poiid/name 及结构化字段 category/rating/average_cost/open_time/distance_m 等，或笔记 content/created_at)、最终推荐(recommendations：poiid/name/引用证据id/constraint_gaps)。

【逐约束判定证据状态】对 constraints 里每一条给出 status ∈ {sufficient 充分, partial 部分, missing 缺失, conflicting 冲突, expired 过期}：
- 证据必须对应**正确的 POI/分店**且可追溯(引用 id 能在 all_observed_evidence 找到且属于该 POI)；张冠李戴按 missing/错接处理。
- 价格/评分/设施/营业时间等→需直接字段或明确内容；安静/体验好/回头客多等→需相应 UGC，不能用高评分替代。
- 时效：created_at 晚于查询 time 的笔记不能作为当时证据→expired；静态 open_time 不能证明"现在/此刻实时有位、营业中"；直线 distance_m 不能证明"路线时长/通勤时间"。
- 比较/最高级("更高/最近/特别多/最好"等)→需合理候选集合与可比证据；只核验一个候选就下最优结论→comparison_single_candidate=true。
- 有限 top-k 或一次空结果只能证明"本次检索没找到"，不能证明"全局不存在"；据此下的否定/最优结论属证据不足。

【打分 0–5(整数)，严格锚定，先逐约束判定再打分；不得因"已找到相关 POI"或"诚实声明了缺口"而抬分】
- 5：全部实际需求均有对象正确、时效有效、可追溯的充分证据；需要比较时覆盖与可比字段充分；推荐及强结论均无证据缺口。
- 4：所有刚性要求和主要柔性偏好均逐项有直接支持，证据对象与适用范围明确；至多存在不影响任何需求判断的次要呈现不清。
- 3：决定推荐合法性与主要结论的证据完整，只缺少次要柔性偏好的补充支持；不存在未核实的关键条件、错接对象或未解决反证。
- 2：刚性要求已有支持，但主要柔性偏好仅部分证实，或比较/强结论覆盖不足。
- 1：至少一个决定推荐合法性的刚性要求仍缺证据或有未解决冲突，其余证据只能支持部分需求；诚实声明缺口不能补足本维度。
- 0：多个刚性要求缺证据，或核心证据对象错误/过期/被编造，或几乎没有任务所需的有效证据(0 用于完全无有效支持)。

【反通胀硬性规则，违反判为无效输出并重打】(hard_lacking = 刚性约束 status∈{missing,conflicting,expired} 的条数；hard_partial = 刚性约束 status=partial 的条数)
1. hard_lacking≥2 → 分数必须 0；hard_lacking==1 → 分数 ≤1；hard_partial≥1(且 hard_lacking=0)→ 分数 ≤2(刚性仅部分支持)；刚性全部 sufficient → 分数 ≥2。
2. comparison_single_candidate=true → 分数 ≤2。
3. fabricated_or_wrong_object=true(核心证据错接/过期/编造) → 分数 ≤1。
4. 分数≥4 → 必须刚性与柔性约束全部 sufficient、且 comparison_single_candidate=false、fabricated_or_wrong_object=false。
5. constraint_evidence 必须覆盖 constraints 里每一个 cid，否则无效。
6. 不得因最终答案是否合格、文案是否漂亮而加减分；只看证据。

只输出合法 JSON，不要多余文字：
{"task_id":"...","constraint_evidence":[{"cid":"c1","status":"sufficient","note":"简短依据/缺什么"}],"comparison_single_candidate":false,"fabricated_or_wrong_object":false,"evidence_score":3,"reason":"一句话总评"}'''


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
            {"cid": c.get("cid"), "text": c.get("text"), "h": c.get("h"), "evidence_type": c.get("evidence_type")}
            for c in (rec.get("constraints") or [])
        ]
    return out


def build_payload(record: Dict[str, Any], task_id: str, constraints: list) -> Dict[str, Any]:
    answer = record.get("answer") or {}
    inp = record.get("input") or {}
    _, by_evidence = evidence_catalog(record)
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
        "all_observed_evidence": list(by_evidence.values()),
        "recommendations": recs,
        "final": {"n_recommendations": len(recs), "no_exact_match": answer.get("no_exact_match")},
    }


def validate(value: Any, cid_h: Dict[str, str]) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("not an object")
    score = value.get("evidence_score")
    if not isinstance(score, int) or isinstance(score, bool) or not 0 <= score <= 5:
        raise ValueError("evidence_score must be int 0..5")
    ce = value.get("constraint_evidence")
    if not isinstance(ce, list):
        raise ValueError("constraint_evidence must be a list")
    cmap: Dict[str, str] = {}
    for e in ce:
        if not isinstance(e, dict):
            raise ValueError("constraint_evidence items must be objects")
        cid = e.get("cid"); st = e.get("status")
        if cid not in cid_h:
            raise ValueError(f"unknown cid {cid}")
        if st not in STATUSES:
            raise ValueError(f"bad status {st}")
        cmap[cid] = st
    if set(cmap) != set(cid_h):
        raise ValueError("constraint_evidence must cover every cid exactly")
    comp = value.get("comparison_single_candidate")
    fab = value.get("fabricated_or_wrong_object")
    if not isinstance(comp, bool) or not isinstance(fab, bool):
        raise ValueError("comparison_single_candidate / fabricated_or_wrong_object must be bool")
    reason = value.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason must be nonempty str")
    hd = sum(1 for c, h in cid_h.items() if h == "刚性" and cmap.get(c) in ("missing", "conflicting", "expired"))
    hp = sum(1 for c, h in cid_h.items() if h == "刚性" and cmap.get(c) == "partial")
    sd = sum(1 for c, h in cid_h.items() if h != "刚性" and cmap.get(c) != "sufficient")
    if hd >= 2 and score != 0:
        raise ValueError(">=2 hard constraints lacking evidence forces score 0")
    if hd == 1 and score > 1:
        raise ValueError("1 hard constraint lacking evidence caps score at 1")
    if hp >= 1 and hd == 0 and score > 2:
        raise ValueError("a hard constraint only partially evidenced caps score at 2")
    if fab and score > 1:
        raise ValueError("fabricated/wrong-object caps score at 1")
    if comp and score > 2:
        raise ValueError("comparison_single_candidate caps score at 2")
    if hd == 0 and hp == 0 and not fab and score < 2:
        raise ValueError("all hard constraints sufficient -> score>=2")
    if score <= 1 and not (hd >= 1 or fab):
        raise ValueError("score<=1 requires a hard constraint lacking evidence or fabrication")
    if score >= 4 and (hd or hp or sd or comp or fab):
        raise ValueError("score>=4 requires all constraints sufficient, no single-candidate comparison, no fabrication")
    return {
        "task_id": value.get("task_id"), "evidence_score": score,
        "constraint_evidence": ce, "comparison_single_candidate": comp,
        "fabricated_or_wrong_object": fab, "hard_deficient": hd, "hard_partial": hp, "soft_deficient": sd,
        "reason": reason.strip(),
    }


def call_judge(args, api_key: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    cid_h = {c["cid"]: c.get("h") for c in payload.get("constraints") or []}
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
    last = None
    for attempt in range(args.judge_retries):
        body = {"model": args.judge_model, "messages": messages,
                "temperature": args.judge_temperature, "max_tokens": args.judge_max_tokens}
        request = urllib.request.Request(
            args.judge_url, data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + api_key}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=args.judge_timeout) as response:
                raw = json.loads(response.read().decode("utf-8"))
            judgment = validate(parse_json(raw["choices"][0]["message"]["content"]), cid_h)
            judgment["task_id"] = payload["task_id"]
            return {"status": "completed", "judgment": judgment, "usage": raw.get("usage") or {}}
        except urllib.error.HTTPError as exc:
            last = exc
            if attempt + 1 < args.judge_retries:
                time.sleep(min(2 ** attempt, 8))
        except Exception as exc:
            last = exc
            if attempt + 1 < args.judge_retries:
                messages.append({"role": "user", "content": f"上一输出无效：{exc}。只返回符合schema的JSON，constraint_evidence 必须覆盖全部 cid。"})
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
    parser.add_argument("--judge-max-tokens", type=int, default=3072)
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
                    counts[result["judgment"]["evidence_score"]] += 1
                else:
                    counts["failed"] += 1
                if index % 100 == 0:
                    print(f"{index}/{len(pending)}", flush=True)
    judged = sum(counts[s] for s in range(0, 6))
    mean = round(sum(s * counts[s] for s in range(0, 6)) / judged, 4) if judged else None
    print(json.dumps({"label": args.label, "judged": judged, "failed": counts["failed"],
                      "score_dist": {str(s): counts[s] for s in range(0, 6)}, "mean_score": mean}, ensure_ascii=False))


if __name__ == "__main__":
    main()
