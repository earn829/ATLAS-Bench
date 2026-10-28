#!/usr/bin/env python3
"""Score Qoder blind-audit verdicts with the published ATLAS-Bench answer metrics.

Computes CSR, CR, Success@k and Recs/query exactly as the main results table does
(judge/analyze_regional_results.py): GOOD = {qualified, acceptable}, constraint
denominator = judged_rows x n_constraints, Success@k and Recs/query averaged over
the should-answer subset (payload "satisfiable" is not False).

Validates the verdict contract first and refuses to score on any violation. Tasks
whose verdict rows are not yet complete are excluded from the metrics and reported
as a coverage shortfall, so interim numbers stay honest. Read-only unless writing
summaries is enabled.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_JUDGE_DIR = ROOT / "judge"
GOOD = {"qualified", "acceptable"}
VERDICTS = GOOD | {"unacceptable"}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def discover_labels(judge_dir: Path) -> List[str]:
    if not judge_dir.is_dir():
        return []
    return sorted(
        entry.name for entry in judge_dir.iterdir()
        if entry.is_dir() and list((entry / "inputs").glob("chunk_*.json"))
    )


def load_inputs(label_dir: Path) -> Dict[str, Dict[str, Any]]:
    tasks: Dict[str, Dict[str, Any]] = {}
    chunks = sorted((label_dir / "inputs").glob("chunk_*.json"))
    if not chunks:
        raise ValueError(f"{label_dir}: no inputs/chunk_*.json — run build_qoder_judge_inputs.py first")
    for path in chunks:
        for payload in json.loads(path.read_text(encoding="utf-8")):
            tid = payload["task_id"]
            if tid in tasks:
                raise ValueError(f"{label_dir}: duplicate task_id {tid!r} across input chunks")
            tasks[tid] = {
                "chunk": path.stem,
                "n_constraints": int(payload["n_constraints"]),
                "satisfiable": payload.get("satisfiable"),
                "recommendations": [(int(r["rank"]), r["poiid"]) for r in payload["recommendations"]],
            }
    return tasks


def load_verdicts(label_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted((label_dir / "verdicts").glob("chunk_*.jsonl")):
        for row in read_jsonl(path):
            row = dict(row)
            row["_chunk"] = path.stem
            rows.append(row)
    return rows


def validate(tasks: Dict[str, Dict[str, Any]], rows: List[Dict[str, Any]]) -> List[str]:
    errors: List[str] = []
    seen = set()
    for row in rows:
        tid, rank = row.get("task_id"), row.get("rank")
        where = f"{row.get('_chunk')}:{tid}#{rank}"
        task = tasks.get(tid)
        if task is None:
            errors.append(f"{where}: task_id absent from the blind inputs")
            continue
        if row.get("_chunk") != task["chunk"]:
            errors.append(f"{where}: verdict written to {row.get('_chunk')} but the query lives in {task['chunk']}")
        if (tid, rank) in seen:
            errors.append(f"{where}: duplicate verdict row")
            continue
        seen.add((tid, rank))
        known = dict(task["recommendations"])
        if rank not in known:
            errors.append(f"{where}: rank not among the input recommendations {sorted(known)}")
            continue
        if row.get("poiid") != known[rank]:
            errors.append(f"{where}: poiid {row.get('poiid')!r} != input {known[rank]!r}")
        if row.get("verdict") not in VERDICTS:
            errors.append(f"{where}: verdict must be one of {sorted(VERDICTS)}, got {row.get('verdict')!r}")
        met = row.get("constraints_met")
        n = task["n_constraints"]
        if isinstance(met, bool) or not isinstance(met, int):
            errors.append(f"{where}: constraints_met must be an int, got {met!r}")
        elif not 0 <= met <= n:
            errors.append(f"{where}: constraints_met={met} outside 0..{n}")
        elif row.get("verdict") == "qualified" and met != n:
            errors.append(f"{where}: qualified requires constraints_met == n_constraints ({n}), got {met}")
        reason = row.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            errors.append(f"{where}: reason must be a non-empty string")
    return errors


def complete_tasks(tasks: Dict[str, Dict[str, Any]], rows: List[Dict[str, Any]]) -> Tuple[set, Dict[str, List[Dict[str, Any]]]]:
    by_task: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_task.setdefault(row.get("task_id"), []).append(row)
    done = {
        tid for tid, task in tasks.items()
        if len(by_task.get(tid, [])) == len(task["recommendations"])
        and sorted(int(r["rank"]) for r in by_task.get(tid, [])) == [rank for rank, _ in task["recommendations"]]
    }
    return done, by_task


def score(tasks: Dict[str, Dict[str, Any]], rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    done, by_task = complete_tasks(tasks, rows)
    usable = [row for tid, rs in by_task.items() if tid in done for row in rs]
    scored = {tid: tasks[tid] for tid in done}

    denominator = 0
    csr_num = 0
    cr_num = 0
    verdict_counts = {"qualified": 0, "acceptable": 0, "unacceptable": 0}
    met_by_verdict: Dict[str, List[int]] = {key: [] for key in verdict_counts}
    for row in usable:
        n = scored[row["task_id"]]["n_constraints"]
        met = int(row.get("constraints_met") or 0)
        denominator += n
        cr_num += met
        verdict = row["verdict"]
        verdict_counts[verdict] += 1
        met_by_verdict[verdict].append(met)
        if verdict in GOOD:
            csr_num += met

    should = {tid: task for tid, task in scored.items() if task["satisfiable"] is not False}
    success = [int(any(r["verdict"] in GOOD for r in by_task.get(tid, []))) for tid in should]
    recs = [len(scored[tid]["recommendations"]) for tid in should]
    expected_rows = sum(len(task["recommendations"]) for task in tasks.values())

    return {
        "queries": len(scored),
        "queries_in_inputs": len(tasks),
        "should_answer_queries": len(should),
        "unsatisfiable_queries": len(scored) - len(should),
        "judged_rows": len(usable),
        "written_rows": len(rows),
        "expected_rows": expected_rows,
        "coverage": round(len(usable) / expected_rows, 6) if expected_rows else None,
        "constraint_denominator": denominator,
        "csr": round(100 * csr_num / denominator, 4) if denominator else None,
        "cr": round(100 * cr_num / denominator, 4) if denominator else None,
        "success_at_k": round(100 * statistics.mean(success), 4) if success else None,
        "recs_per_query": round(statistics.mean(recs), 4) if recs else None,
        "verdict_counts": verdict_counts,
        "avg_constraints_met_by_verdict": {
            key: round(statistics.mean(values), 4) if values else None
            for key, values in met_by_verdict.items()
        },
        "avg_constraints_per_query": round(
            statistics.mean([task["n_constraints"] for task in scored.values()]), 4
        ) if scored else None,
    }


def score_label(label: str, judge_dir: Path) -> Dict[str, Any]:
    label_dir = judge_dir / label
    tasks = load_inputs(label_dir)
    rows = load_verdicts(label_dir)
    errors = validate(tasks, rows)
    if errors:
        raise ValueError(label, errors)
    return score(tasks, rows)


def markdown_table(results: Dict[str, Dict[str, Any]]) -> str:
    lines = [
        "| Label | Queries | Should | Rows | Coverage | CSR | CR | Success@k | Recs/query |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    ordered = sorted(results.items(), key=lambda kv: -(kv[1]["csr"] or 0))
    for label, r in ordered:
        lines.append(
            "| {label} | {q}/{qi} | {s} | {jr}/{er} | {cov} | {csr} | {cr} | {sk} | {rq} |".format(
                label=label, q=r["queries"], qi=r["queries_in_inputs"], s=r["should_answer_queries"],
                jr=r["judged_rows"], er=r["expected_rows"],
                cov="1.000" if r["coverage"] == 1 else "%.3f" % (r["coverage"] or 0),
                csr="%.2f" % r["csr"] if r["csr"] is not None else "-",
                cr="%.2f" % r["cr"] if r["cr"] is not None else "-",
                sk="%.2f" % r["success_at_k"] if r["success_at_k"] is not None else "-",
                rq="%.2f" % r["recs_per_query"] if r["recs_per_query"] is not None else "-",
            )
        )
    return "\n".join(lines) + "\n"


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--judge-dir", type=Path, default=DEFAULT_JUDGE_DIR,
                        help=f"Judge root holding <label>/inputs and <label>/verdicts (default: {DEFAULT_JUDGE_DIR})")
    parser.add_argument("--no-write", action="store_true",
                        help="Score and print only; do not write summary_published.json or _published_metrics.md")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of the table")
    parser.add_argument("labels", nargs="*", help="Judge labels (default: every label with inputs)")
    args = parser.parse_args(argv)

    judge_dir = args.judge_dir.resolve()
    labels = args.labels or discover_labels(judge_dir)
    if not labels:
        print(f"No judge labels with inputs under {judge_dir}.", file=sys.stderr)
        return 1

    results: Dict[str, Dict[str, Any]] = {}
    failures = 0
    for label in labels:
        try:
            result = score_label(label, judge_dir)
        except ValueError as exc:
            if isinstance(exc.args[0], str) and len(exc.args) == 2 and isinstance(exc.args[1], list):
                label_name, errors = exc.args
                failures += 1
                print(f"{label_name}: CONTRACT VIOLATIONS ({len(errors)}), not scored:", file=sys.stderr)
                for item in errors[:20]:
                    print(f"  - {item}", file=sys.stderr)
                if len(errors) > 20:
                    print(f"  ... {len(errors) - 20} more", file=sys.stderr)
                continue
            print(str(exc), file=sys.stderr)
            return 1
        results[label] = result
        if not args.no_write:
            out = judge_dir / label / "summary_published.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if not results:
        return 2

    if args.json:
        print(json.dumps({"judge_dir": str(judge_dir), "labels": results}, ensure_ascii=False, indent=2))
    else:
        print(markdown_table(results))
        short = [label for label, r in results.items() if r["coverage"] != 1]
        if short:
            print(f"\nWARNING: incomplete coverage for {', '.join(short)} — "
                  "metrics cover only fully judged queries.", file=sys.stderr)
        if not args.no_write:
            table_path = judge_dir / "_published_metrics.md"
            table_path.write_text(markdown_table(results), encoding="utf-8")
            print(f"Wrote {table_path}", file=sys.stderr)
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
