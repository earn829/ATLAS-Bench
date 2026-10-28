"""Collapse a resumed run into one record per task and emit the run summary.

Resume passes append, so a task may appear several times: a failure followed by a
later success. Keeping the wrong line would silently corrupt every downstream
metric, so the preference rule is explicit and tested.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional


def read_jsonl(path: Path) -> List[str]:
    """Read physical JSONL lines.

    ``str.splitlines`` also splits on Unicode line separators that can legally appear
    inside JSON string values, so it would corrupt valid records.
    """
    with path.open(encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle]


def dedupe_records(raw_lines: List[str]) -> tuple[List[Dict[str, Any]], Counter, Counter]:
    """Return one record per task_id, preferring completed over failed.

    Ties are broken by file order (the later line wins), so the last resume pass
    has the final say.
    """
    best: Dict[str, Dict[str, Any]] = {}
    attempts: Counter = Counter()
    parse_errors: Counter = Counter()
    for line_number, line in enumerate(raw_lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            parse_errors["line_%d" % line_number] += 1
            continue
        task_id = record.get("task_id")
        if not isinstance(task_id, str):
            parse_errors["line_%d" % line_number] += 1
            continue
        attempts[task_id] += 1
        previous = best.get(task_id)
        if previous is None:
            best[task_id] = record
            continue
        previous_ok = previous.get("status") == "completed"
        current_ok = record.get("status") == "completed"
        if current_ok or not previous_ok:
            best[task_id] = record
    return list(best.values()), attempts, parse_errors


def _usage_of(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    events = (record.get("metrics") or {}).get("usage_events")
    return events if isinstance(events, list) else []


def summarize(
    records: List[Dict[str, Any]],
    *,
    raw_line_count: int,
    attempts: Counter,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    completed = [r for r in records if r.get("status") == "completed"]
    failed = [r for r in records if r.get("status") != "completed"]

    distribution: Counter = Counter()
    for record in completed:
        recommendations = (record.get("answer") or {}).get("recommendations") or []
        distribution[len(recommendations)] += 1
    recommendation_total = sum(
        count * size for size, count in distribution.items()
    )

    prompt_tokens = completion_tokens = reasoning_tokens = 0
    tool_calls = 0
    elapsed: List[float] = []
    max_completion_per_call = 0
    for record in completed:
        metrics = record.get("metrics") or {}
        tool_calls += int(metrics.get("tool_calls") or 0)
        if isinstance(metrics.get("elapsed_seconds"), (int, float)):
            elapsed.append(float(metrics["elapsed_seconds"]))
        for usage in _usage_of(record):
            prompt_tokens += int(usage.get("prompt_tokens") or 0)
            call_completion = int(usage.get("completion_tokens") or 0)
            completion_tokens += call_completion
            max_completion_per_call = max(max_completion_per_call, call_completion)
            details = usage.get("completion_tokens_details") or {}
            reasoning_tokens += int(details.get("reasoning_tokens") or 0)
    elapsed.sort()

    judge = Counter()
    for record in completed:
        verdict = record.get("programmatic_judge")
        if not isinstance(verdict, dict):
            continue
        judge["evaluated_answers"] += 1
        judge["passed_answers"] += int(bool(verdict.get("passed")))
        judge["no_exact_match_consistent"] += int(
            bool((verdict.get("no_exact_match") or {}).get("consistent"))
        )
        ownership = verdict.get("evidence_ownership") or {}
        for target, source in (
            ("evidence_reference_count", "reference_count"),
            ("evidence_known_count", "known_count"),
            ("evidence_correct_owner_count", "correct_owner_count"),
            ("evidence_wrong_owner_count", "wrong_owner_count"),
            ("evidence_unknown_count", "unknown_count"),
        ):
            judge[target] += int(ownership.get(source) or 0)

    def ratio(numerator: int, denominator: int) -> Optional[float]:
        return round(numerator / denominator, 6) if denominator else None

    error_types = Counter(
        (record.get("error") or {}).get("type") or "unknown" for record in failed
    )
    retried = sum(1 for count in attempts.values() if count > 1)

    return {
        **config,
        "raw_lines": raw_line_count,
        "records": len(records),
        "completed": len(completed),
        "failed": len(failed),
        "tasks_with_multiple_attempts": retried,
        "max_attempts_per_task": max(attempts.values()) if attempts else 0,
        "failure_types": dict(error_types),
        "recommendation_distribution": {
            str(size): distribution[size] for size in sorted(distribution)
        },
        "recommendation_mean": (
            round(recommendation_total / len(completed), 4) if completed else None
        ),
        "over5_count": sum(c for s, c in distribution.items() if s > 5),
        "usage": {
            "prompt_tokens_total": prompt_tokens,
            "completion_tokens_total": completion_tokens,
            "reasoning_tokens_total": reasoning_tokens,
            "prompt_avg_per_task": round(prompt_tokens / len(completed), 1)
            if completed
            else None,
            "completion_avg_per_task": round(completion_tokens / len(completed), 1)
            if completed
            else None,
            "max_completion_tokens_single_call": max_completion_per_call,
            "tool_calls_avg": round(tool_calls / len(completed), 3) if completed else None,
            "elapsed_median_s": round(elapsed[len(elapsed) // 2], 1) if elapsed else None,
        },
        "programmatic_judge": {
            **{key: judge[key] for key in sorted(judge)},
            "answer_pass_rate": ratio(judge["passed_answers"], judge["evaluated_answers"]),
            "no_exact_match_consistency_rate": ratio(
                judge["no_exact_match_consistent"], judge["evaluated_answers"]
            ),
            "evidence_existence_accuracy": ratio(
                judge["evidence_known_count"], judge["evidence_reference_count"]
            ),
            "evidence_ownership_accuracy": ratio(
                judge["evidence_correct_owner_count"], judge["evidence_reference_count"]
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="run output jsonl (may contain resume appends)")
    parser.add_argument("--tasks", default=str(Path(__file__).resolve().parents[1] / "query" / "subset_1200_agent_input.json"))
    parser.add_argument("--output", help="default: <input stem>.final.jsonl next to the input")
    parser.add_argument("--config-json", default="{}", help="run config recorded verbatim in the summary")
    args = parser.parse_args()

    source = Path(args.input)
    raw_lines = read_jsonl(source)
    records, attempts, parse_errors = dedupe_records(raw_lines)
    if parse_errors:
        raise ValueError("输入存在无法解析的行：%s" % dict(parse_errors))

    order = {
        record["id"]: index
        for index, record in enumerate(json.load(open(args.tasks, encoding="utf-8"))["records"])
    }
    records.sort(key=lambda r: order.get(r.get("task_id"), len(order)))
    missing = sorted(set(order) - {r.get("task_id") for r in records})

    target = Path(args.output) if args.output else source.with_name(
        source.name.replace(".jsonl", "") + ".final.jsonl"
    )
    if target.resolve() == source.resolve():
        raise ValueError("输出不能覆盖输入")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = summarize(
        records,
        raw_line_count=len(raw_lines),
        attempts=attempts,
        config=json.loads(args.config_json),
    )
    summary["missing_task_ids"] = missing
    summary_path = target.with_name(target.name + ".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "final": str(target),
                "summary": str(summary_path),
                "records": summary["records"],
                "completed": summary["completed"],
                "failed": summary["failed"],
                "missing": len(missing),
                "failure_types": summary["failure_types"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
