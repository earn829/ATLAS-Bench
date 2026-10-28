"""Build per-query audit payloads for the Qoder full-judge pass.

For every finalized 1200-query run, emit chunked input files under
judge/<model>/inputs/ containing only blind, judge-relevant fields:
query, context, annotated constraints, and each recommendation's
compact_answer (cited_evidence + observed_facts). Reference answers,
target POIs and model identity are excluded.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "scripts"))
sys.path.insert(0, str(_HERE))

from finalize_run import dedupe_records, read_jsonl  # noqa: E402
from paired_acceptability_judge import compact_answer  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "query" / "subset_1200.json"
JUDGE_DIR = ROOT / "judge"

MODELS: Dict[str, str] = {
    "qwen35_4b_nothink": "qwen3.5-4B-local-nothinking-topk20-reclimit10",
    "qwen35_4b_think": "qwen3.5-4B-local-thinking-rp115-topk20-reclimit10",
    "qwen36_a3b_nothink": "qwen3.6-35B-A3B-nothinking-topk20-reclimit10",
    "qwen36_a3b_think": "qwen3.6-35B-A3B-thinking-topk20-reclimit10",
    "qwen38_27b_nothink": "qwen3.8-27B-local-nothinking-topk20-reclimit10",
    "qwen38_27b_think": "qwen3.8-27B-local-thinking-rp105-topk20-reclimit10",
    "qwen38_max": "qwen3.8-max-hosted-default-thinking-topk20-reclimit10",
    "deepseek_v41_flash": "deepseek-v4-1-flash-hosted-topk20-reclimit10",
    "claude5_sonnet": "bedrock-claude-sonnet-5-hosted-default-thinking-cache-topk20-reclimit10",
    "gpt56_luna": "azure-gpt-5-6-luna-hosted-default-thinking-topk20-reclimit10",
    "gpt6_luna": "gpt-6-luna-hosted-xhigh-topk20-reclimit10",
    "glm53": "glm-5-3-hosted-default-thinking-topk20-reclimit10",
    "kimi_k3": "kimi-k3-hosted-default-thinking-topk20-reclimit10",
    "gemini38_flash": "gemini-3-8-flash-hosted-default-thinking-topk20-reclimit10",
}


def load_tasks(path: Path = SOURCE) -> Dict[str, Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, list):
        raise ValueError(f"{path} must contain a raw annotation array")
    tasks: Dict[str, Dict[str, Any]] = {}
    for index, item in enumerate(raw, 1):
        if not isinstance(item, dict) or not isinstance(item.get("record"), dict):
            raise ValueError(f"{path}: annotation {index} must contain a record object")
        rec = item["record"]
        task_id = rec.get("id")
        if not isinstance(task_id, str):
            raise ValueError(f"{path}: annotation {index} must have a string record.id")
        if task_id in tasks:
            raise ValueError(f"{path}: duplicate task ID {task_id!r}")
        tasks[task_id] = {
            "id": rec["id"],
            "query": rec.get("query"),
            "context": rec.get("context") or {},
            "constraints": rec.get("constraints") or [],
            "structure": rec.get("structure") or {},
            "satisfiable": (rec.get("answer") or {}).get("satisfiable"),
        }
    return tasks


def slim_constraint(c: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "cid": c.get("cid"),
        "text": c.get("text"),
        "h": c.get("h"),
        "s": c.get("s"),
        "evidence_type": c.get("evidence_type"),
        "expected_predicate": c.get("expected_predicate"),
    }


def build_payload(task: Dict[str, Any], record: Dict[str, Any]) -> Dict[str, Any]:
    recs = compact_answer(record)
    return {
        "task_id": task["id"],
        "query": task["query"],
        "context": task["context"],
        "constraints": [slim_constraint(c) for c in task["constraints"]],
        "n_constraints": len(task["constraints"]),
        "satisfiable": task["satisfiable"],
        "recommendations": recs,
    }


def chunk_by_budget(payloads: List[Dict[str, Any]], budget_chars: int) -> List[List[Dict[str, Any]]]:
    chunks: List[List[Dict[str, Any]]] = []
    cur: List[Dict[str, Any]] = []
    cur_len = 0
    for p in payloads:
        size = len(json.dumps(p, ensure_ascii=False))
        if cur and cur_len + size > budget_chars:
            chunks.append(cur)
            cur, cur_len = [], 0
        cur.append(p)
        cur_len += size
    if cur:
        chunks.append(cur)
    return chunks


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, help="Finalized JSONL run for a single label")
    parser.add_argument("--label", help="Output label for --run (one safe path component)")
    parser.add_argument("--tasks", type=Path, default=SOURCE)
    parser.add_argument("--judge-dir", type=Path, default=JUDGE_DIR)
    parser.add_argument("--runs-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--budget-chars", type=int, help="Chunk character budget (default: 60000)")
    parser.add_argument(
        "models", nargs="*", metavar="MODEL",
        help="Known batch model selectors, optionally preceded by a legacy character budget",
    )
    args = parser.parse_intermixed_args(argv)

    legacy_budget = None
    if args.models:
        try:
            legacy_budget = int(args.models[0])
        except ValueError:
            pass
        else:
            args.models = args.models[1:]
    if args.budget_chars is not None and legacy_budget is not None and args.budget_chars != legacy_budget:
        parser.error("--budget-chars conflicts with the positional budget")
    budget = args.budget_chars if args.budget_chars is not None else legacy_budget
    if budget is None:
        budget = 60000
    if budget <= 0:
        parser.error("the character budget must be positive")

    if (args.run is None) != (args.label is None):
        parser.error("--run and --label must be supplied together")
    if args.run is not None:
        if args.models:
            parser.error("--run/--label cannot be mixed with batch model selectors")
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.label) is None:
            parser.error("--label must be one safe path component: [A-Za-z0-9][A-Za-z0-9_.-]*")
        runs = [(args.label, args.run.resolve())]
    else:
        unknown = [model for model in args.models if model not in MODELS]
        if unknown:
            parser.error(f"unknown model selectors: {', '.join(unknown)}")
        runs_root = args.runs_root.resolve()
        runs = [
            (model, runs_root / rundir / "subset_1200.final.jsonl")
            for model, rundir in MODELS.items()
            if not args.models or model in args.models
        ]

    for _, run_path in runs:
        if not run_path.is_file():
            raise FileNotFoundError(f"Run file does not exist: {run_path}")
    tasks = load_tasks(args.tasks.resolve())
    order = list(tasks.keys())
    prepared = []
    for model, run_path in runs:
        records, _, parse_errors = dedupe_records(read_jsonl(run_path))
        if parse_errors:
            raise ValueError(f"{run_path} has unparseable lines: {dict(parse_errors)}")
        by_id = {r["task_id"]: r for r in records if r.get("status") == "completed"}
        if not by_id:
            raise ValueError(f"{run_path} has no completed records")
        unknown_ids = sorted(set(by_id) - set(tasks))
        if unknown_ids:
            raise ValueError(f"{run_path}: completed task IDs not found in annotations: {unknown_ids}")
        payloads = [build_payload(tasks[tid], by_id[tid]) for tid in order if tid in by_id]
        chunks = chunk_by_budget(payloads, budget)
        prepared.append((model, payloads, chunks))

    judge_dir = args.judge_dir.resolve()
    report = []
    for model, payloads, chunks in prepared:
        in_dir = judge_dir / model / "inputs"
        (judge_dir / model / "verdicts").mkdir(parents=True, exist_ok=True)
        in_dir.mkdir(parents=True, exist_ok=True)
        for old in in_dir.glob("chunk_*.json"):
            old.unlink()
        for i, ch in enumerate(chunks, 1):
            (in_dir / f"chunk_{i:02d}.json").write_text(
                json.dumps(ch, ensure_ascii=False), encoding="utf-8"
            )
        total_recs = sum(len(p["recommendations"]) for p in payloads)
        report.append({
            "model": model,
            "queries": len(payloads),
            "recommendations": total_recs,
            "chunks": len(chunks),
            "budget_chars": budget,
        })
        print(json.dumps(report[-1], ensure_ascii=False), flush=True)
    (judge_dir / "_manifest.json").write_text(
        json.dumps({"budget_chars": budget, "models": report}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
