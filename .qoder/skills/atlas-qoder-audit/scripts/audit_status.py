#!/usr/bin/env python3
"""Report Qoder blind-audit progress for one or more judge labels.

Reads <judge-dir>/<label>/inputs/chunk_NN.json and <judge-dir>/<label>/verdicts/
chunk_NN.jsonl, then reports per-chunk completeness, the chunks still pending
adjudication, and any contract anomaly (row in the wrong chunk, duplicate
(task_id, rank), unknown task_id, verdict outside the allowed set). Read-only:
never writes, never calls a model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_JUDGE_DIR = ROOT / "judge"
VERDICTS = {"qualified", "acceptable", "unacceptable"}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def discover_labels(judge_dir: Path) -> List[str]:
    return sorted(
        entry.name
        for entry in judge_dir.iterdir()
        if entry.is_dir() and list((entry / "inputs").glob("chunk_*.json"))
    ) if judge_dir.is_dir() else []


def chunk_status(label_dir: Path) -> Dict[str, Any]:
    inputs = sorted((label_dir / "inputs").glob("chunk_*.json"))
    if not inputs:
        raise ValueError(f"{label_dir}: no inputs/chunk_*.json — run build_qoder_judge_inputs.py first")
    verdict_dir = label_dir / "verdicts"
    chunks = []
    anomalies: List[str] = []
    totals = {"queries": 0, "expected_rows": 0, "written_rows": 0, "complete_chunks": 0}
    for path in inputs:
        payloads = json.loads(path.read_text(encoding="utf-8"))
        expected = {p["task_id"]: len(p["recommendations"]) for p in payloads}
        poiids = {p["task_id"]: {int(r["rank"]): r["poiid"] for r in p["recommendations"]} for p in payloads}
        n_constraints = {p["task_id"]: int(p["n_constraints"]) for p in payloads}
        vpath = verdict_dir / f"{path.stem}.jsonl"
        rows = read_jsonl(vpath) if vpath.is_file() else None
        totals["queries"] += len(payloads)
        totals["expected_rows"] += sum(expected.values())
        state = {"chunk": path.stem, "queries": len(payloads),
                 "expected_rows": sum(expected.values())}
        if rows is None:
            state.update(status="missing", written_rows=0)
            chunks.append(state)
            continue
        totals["written_rows"] += len(rows)
        state["written_rows"] = len(rows)
        before = len(anomalies)
        seen = set()
        for row in rows:
            tid = row.get("task_id")
            rank = row.get("rank")
            key = (tid, rank)
            if tid not in expected:
                anomalies.append(f"{path.stem}: task_id {tid!r} is not in this input chunk")
            elif key in seen:
                anomalies.append(f"{path.stem}: duplicate row {key}")
            else:
                seen.add(key)
                if rank not in poiids[tid]:
                    anomalies.append(f"{path.stem}: {key} rank not among {sorted(poiids[tid])}")
                elif row.get("poiid") != poiids[tid][rank]:
                    anomalies.append(f"{path.stem}: {key} poiid {row.get('poiid')!r} != input {poiids[tid][rank]!r}")
            if row.get("verdict") not in VERDICTS:
                anomalies.append(f"{path.stem}: bad verdict {row.get('verdict')!r} for {key}")
                continue
            met = row.get("constraints_met")
            limit = n_constraints.get(tid, 0)
            if isinstance(met, bool) or not isinstance(met, int) or not 0 <= met <= limit:
                anomalies.append(f"{path.stem}: {key} constraints_met {met!r} outside 0..{limit}")
            elif row["verdict"] == "qualified" and met != limit:
                anomalies.append(f"{path.stem}: {key} qualified with constraints_met {met} != {limit}")
        state["anomalies"] = len(anomalies) - before
        short = sorted(t for t, n in expected.items() if n > 0 and
                       sum(1 for r in rows if r.get("task_id") == t) != n)
        if state["anomalies"]:
            state["status"] = "invalid"
        elif not short and len(rows) == sum(expected.values()):
            state["status"] = "complete"
            totals["complete_chunks"] += 1
        else:
            state["status"] = "incomplete"
            state["incomplete_tasks"] = short[:10]
        chunks.append(state)
    totals.update(chunks=len(inputs), complete=totals["complete_chunks"] == len(inputs),
                  anomalies=anomalies[:20], anomaly_count=len(anomalies))
    return {"totals": totals, "chunks": chunks}


def pending_chunks(status: Dict[str, Any]) -> List[str]:
    return [c["chunk"] for c in status["chunks"] if c["status"] != "complete"]


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--judge-dir", type=Path, default=DEFAULT_JUDGE_DIR,
                        help=f"Judge root holding <label>/inputs and <label>/verdicts (default: {DEFAULT_JUDGE_DIR})")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of text")
    parser.add_argument("--pending", action="store_true",
                        help="Print only the pending chunk stems, one per line (audit worklist)")
    parser.add_argument("labels", nargs="*", help="Judge labels (default: every label with inputs)")
    args = parser.parse_args(argv)

    judge_dir = args.judge_dir.resolve()
    labels = args.labels or discover_labels(judge_dir)
    if not labels:
        print(f"No judge labels with inputs under {judge_dir}.", file=sys.stderr)
        return 1

    report = {}
    for label in labels:
        label_dir = judge_dir / label
        try:
            report[label] = chunk_status(label_dir)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if args.pending:
        for label, status in report.items():
            prefix = f"{label}/" if len(report) > 1 else ""
            for chunk in pending_chunks(status):
                print(f"{prefix}{chunk}")
        return 0

    if args.json:
        print(json.dumps({"judge_dir": str(judge_dir), "labels": report},
                         ensure_ascii=False, indent=2))
        return 0

    for label, status in report.items():
        t = status["totals"]
        state = "COMPLETE" if t["complete"] else "IN PROGRESS"
        print(f"{label}: {state}  chunks {t['complete_chunks']}/{t['chunks']}  "
              f"queries {t['queries']}  rows {t['written_rows']}/{t['expected_rows']}")
        pending = pending_chunks(status)
        if pending:
            shown = ", ".join(pending[:8]) + (" ..." if len(pending) > 8 else "")
            print(f"  pending ({len(pending)}): {shown}")
        if t["anomaly_count"]:
            print(f"  ANOMALIES ({t['anomaly_count']}):")
            for item in t["anomalies"]:
                print(f"    - {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
