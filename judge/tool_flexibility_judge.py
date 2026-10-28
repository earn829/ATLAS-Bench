"""Strategy-adaptation judge (DeepSeek-Pro).

Scores ONE dimension only: how well the agent adapts its retrieval strategy to
tool observations (reformulating queries, switching/combining tools, adjusting
filters, falling back between nearby<->global recall, poiid lookups, recovering
from empty results) versus rigidly repeating calls, ignoring feedback, or giving
up prematurely.

Deliberately anti-inflation: integer 1-5, default 3, deduction/cap rules that are
enforced in validate() (a 5 requires adaptation needed + zero rigidity signals +
>=2 adaptive moves; trivial one-shot queries cap at 3). Final-answer correctness is
explicitly out of scope so a lucky answer with a rigid process scores low.

Judge model: deepseek-v4-pro on the hosted judge gateway (same harness as
judge/factual_accuracy_judge.py).
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

SYSTEM = '''你是严格的「策略调整」评审员。输入包含：一条本地生活检索 query、上下文(time/loc)、一个匿名 agent 的完整工具调用轨迹（逐轮给出：工具名、参数 arguments、该轮 rationale、观测结果数量 result_count/是否 empty、少量结果样例），以及最终答案概要（推荐数、no_exact_match）。

【只评一个维度：策略调整】
即 agent 是否根据每一步观测结果**动态调整检索策略**——改写 query、切换或组合工具、调整过滤条件(半径/品类/价格/评分)、在「附近↔全局召回」之间回退、用 poiid 回查把笔记候选补成结构化 POI、从空结果或不足结果中恢复；而不是机械重复、无视反馈、或过早放弃。

【不要评】最终答案是否正确、推荐是否满足约束、事实是否准确、同一轮并发多少调用(并行度)。这些都不是本维度。只看「策略是否随反馈灵活演进」。

【打分 1–5（整数），严格锚定，默认 3 分；必须先列证据再打分】
- 5（极少给）：任务确需多轮探索(adaptation_needed=true)，展示了 ≥2 处由观测驱动的有效调整（如：附近搜空→改全局召回；品类过宽→加过滤；笔记命中→用 poiid 回查结构字段），且无任何僵化信号。
- 4：有 ≥1 处明确的、由观测驱动的调整，且无僵化信号；但探索不如 5 充分。
- 3（最常见）：策略基本合理但未展示真正的灵活调整——要么任务简单一次到位(adaptation_needed=false)，要么调整平淡/含冗余。competent but unremarkable。
- 2：存在 1 处实质性僵化信号，导致本可找到的答案没找到、或走了明显弯路。
- 1：存在 ≥2 处实质性僵化信号，或全程几乎无调整（单次调用即放弃 / 反复几乎相同的调用 / 完全无视观测）。

【僵化信号 rigidity_signals，必须逐条指出第几轮 + 具体问题】
- 重复或近乎重复的调用（相同工具 + 几乎相同参数），未依据上一次观测改变；
- 观测已表明此路不通（空结果/品类不符/字段缺失），仍沿同一思路继续或直接放弃，未尝试改写、换工具、放宽或收紧条件；
- 该用 A 工具却反复用 B 工具（如需结构化距离/价格却只搜笔记；需笔记佐证体验却只查 POI 结构）；
- 首轮无结果即返回空答案，且存在可行替代路径却未做任何回退或改写；
- 完全无视工具返回，后续调用与前序观测无关。

【反通胀硬性规则，违反将被判为无效输出并要求重打】
1. 「调用多」≠「灵活」；冗余或重复调用只扣分不加分。
2. score=5 必须同时满足：adaptation_needed=true 且 rigidity_signals 为空 且 adaptive_moves ≥2。
3. score=4 必须同时满足：rigidity_signals 为空 且 adaptive_moves ≥1。
4. score≤2 必须给出 ≥1 条 rigidity_signals。
5. 若任务简单、一次正确到位、无需调整：adaptation_needed=false，且分数最高只能给 3（未展示灵活度，不代表差，但本维度不给高分）。
6. 不得因最终答案的好坏加减分；只评过程的策略演进。

只输出合法 JSON，不要多余文字：
{"task_id":"...","adaptation_needed":true,"adaptive_moves":["第2轮:附近搜空→改全局召回"],"rigidity_signals":["第3-4轮:重复几乎相同search_pois未改参数"],"flexibility_score":3,"reason":"一句话总评"}
adaptive_moves 与 rigidity_signals 必须是具体、可核对、带轮次引用的短语；确实没有则给空数组 []。'''


def load_run(path: Path) -> Dict[str, Dict[str, Any]]:
    records, _, parse_errors = dedupe_records(read_jsonl(path))
    if parse_errors:
        raise ValueError("%s 存在无法解析的行：%s" % (path, dict(parse_errors)))
    return {record["task_id"]: record for record in records}


def compact_trajectory(record: Dict[str, Any]) -> list:
    """Per-round tool-call view focused on strategy, not full result payloads."""
    rounds = []
    r = 0
    for event in record.get("trajectory") or []:
        if event.get("event") != "tool_calls":
            continue
        r += 1
        calls = []
        for call in event.get("calls") or []:
            obs = call.get("observation")
            obs = obs if isinstance(obs, dict) else {}
            data = obs.get("data") if isinstance(obs.get("data"), dict) else {}
            results = data.get("results") or data.get("notes") or []
            cnt = data.get("count")
            if not isinstance(cnt, int):
                cnt = len(results) if isinstance(results, list) else 0
            sample = []
            if isinstance(results, list):
                for item in results[:3]:
                    if isinstance(item, dict):
                        nm = item.get("name") or item.get("poi_name")
                        if isinstance(nm, str):
                            sample.append(nm)
            args = call.get("arguments")
            calls.append(
                {
                    "tool": call.get("tool"),
                    "arguments": args if isinstance(args, dict) else {},
                    "rationale": (call.get("rationale") or "")[:220],
                    "obs_ok": bool(obs.get("ok")),
                    "result_count": cnt,
                    "empty": cnt == 0,
                    "sample_results": sample,
                }
            )
        rounds.append({"round": r, "calls": calls})
    return rounds


def validate(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("not an object")
    score = value.get("flexibility_score")
    if not isinstance(score, int) or isinstance(score, bool) or not 1 <= score <= 5:
        raise ValueError("flexibility_score must be int 1..5")
    needed = value.get("adaptation_needed")
    if not isinstance(needed, bool):
        raise ValueError("adaptation_needed must be bool")
    moves = value.get("adaptive_moves")
    signals = value.get("rigidity_signals")
    if not isinstance(moves, list) or not all(isinstance(x, str) and x.strip() for x in moves):
        raise ValueError("adaptive_moves must be list of nonempty str")
    if not isinstance(signals, list) or not all(isinstance(x, str) and x.strip() for x in signals):
        raise ValueError("rigidity_signals must be list of nonempty str")
    reason = value.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason must be nonempty str")
    # anti-inflation caps (mirror the prompt's hard rules)
    if score == 5 and not (needed and not signals and len(moves) >= 2):
        raise ValueError("score=5 requires adaptation_needed, zero rigidity_signals, >=2 adaptive_moves")
    if score == 4 and not (not signals and len(moves) >= 1):
        raise ValueError("score=4 requires zero rigidity_signals and >=1 adaptive_move")
    if score <= 2 and len(signals) < 1:
        raise ValueError("score<=2 requires >=1 rigidity_signal")
    if not needed and score > 3:
        raise ValueError("adaptation_needed=false caps score at 3")
    return {
        "task_id": value.get("task_id"),
        "adaptation_needed": needed,
        "adaptive_moves": moves,
        "rigidity_signals": signals,
        "flexibility_score": score,
        "reason": reason.strip(),
    }


def call_judge(args: argparse.Namespace, api_key: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    last: Exception | None = None
    for attempt in range(args.judge_retries):
        body = {
            "model": args.judge_model,
            "messages": messages,
            "temperature": args.judge_temperature,
            "max_tokens": args.judge_max_tokens,
        }
        request = urllib.request.Request(
            args.judge_url,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + api_key},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=args.judge_timeout) as response:
                raw = json.loads(response.read().decode("utf-8"))
            text = raw["choices"][0]["message"]["content"]
            judgment = validate(parse_json(text))
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


def build_payload(record: Dict[str, Any], task_id: str) -> Dict[str, Any]:
    answer = record.get("answer") or {}
    inp = record.get("input") or {}
    return {
        "task_id": task_id,
        "query": inp.get("query"),
        "context": {k: v for k, v in inp.items() if k != "query"},
        "trajectory": compact_trajectory(record),
        "final": {
            "n_recommendations": len(answer.get("recommendations") or []),
            "no_exact_match": answer.get("no_exact_match"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True)
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
    parser.add_argument("--dry-run", action="store_true", help="build+print payloads, no LLM call")
    args = parser.parse_args()

    run = load_run(Path(args.run))
    order = list(run.keys())
    if args.limit is not None:
        order = order[: args.limit]

    if args.dry_run:
        for task_id in order:
            print(json.dumps(build_payload(run[task_id], task_id), ensure_ascii=False, indent=2))
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
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("status") == "completed":
                done.add(r["task_id"])
    pending = [task_id for task_id in order if task_id not in done]
    print(json.dumps({"selected": len(order), "pending": len(pending), "model": args.judge_model}, ensure_ascii=False), flush=True)

    def work(task_id: str) -> Dict[str, Any]:
        payload = build_payload(run[task_id], task_id)
        result = call_judge(args, api_key, payload)
        result.update({"task_id": task_id, "label": args.label})
        return result

    lock = threading.Lock()
    counts: Counter = Counter()
    with output.open("w" if args.fresh else "a", encoding="utf-8") as handle:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(work, task_id): task_id for task_id in pending}
            for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
                result = future.result()
                with lock:
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    handle.flush()
                if result["status"] == "completed":
                    counts[result["judgment"]["flexibility_score"]] += 1
                else:
                    counts["failed"] += 1
                if index % 50 == 0:
                    print(f"{index}/{len(pending)}", flush=True)
    judged = sum(counts[s] for s in range(1, 6))
    mean = round(sum(s * counts[s] for s in range(1, 6)) / judged, 4) if judged else None
    print(json.dumps({"label": args.label, "judged": judged, "failed": counts["failed"],
                      "score_dist": {str(s): counts[s] for s in range(1, 6)}, "mean_score": mean}, ensure_ascii=False))


if __name__ == "__main__":
    main()
