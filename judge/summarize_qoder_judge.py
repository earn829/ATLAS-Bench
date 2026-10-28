"""Aggregate Qoder full-judge verdicts into per-model + overall summaries.

Reads judge/<model>/verdicts/chunk_*.jsonl, validates completeness against the
input chunks, and computes verdict counts, qualified ratios, success@k, and
constraint-satisfaction statistics (using constraints_met).
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
JUDGE = ROOT / "judge"

MODEL_LABELS = {
    "qwen35_4b_nothink": "Qwen3.5-4B w/o Think",
    "qwen35_4b_think": "Qwen3.5-4B w/ Think",
    "qwen36_a3b_nothink": "Qwen3.6-35B-A3B w/o Think",
    "qwen36_a3b_think": "Qwen3.6-35B-A3B w/ Think",
    "qwen38_27b_nothink": "Qwen3.8-27B w/o Think",
    "qwen38_27b_think": "Qwen3.8-27B w/ Think",
    "qwen38_max": "Qwen3.8-Max (Max)",
    "deepseek_v41_flash": "DeepSeek-V4.1-Flash (Thinking)",
    "claude5_sonnet": "Claude-5-Sonnet (Adaptive)",
    "gpt56_luna": "GPT-5.6-Luna (Medium)",
    "gpt6_luna": "GPT-6-Luna (XHigh)",
    "glm53": "GLM-5.3 (Max)",
    "kimi_k3": "Kimi-K3 (Max)",
    "gemini38_flash": "Gemini-3.8-Flash (Hidden Think)",
}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def _input_chunks(model: str, judge_dir: Path) -> List[Path]:
    model_dir = judge_dir / model
    guidance = "Run build_qoder_judge_inputs.py and judge the input chunks first."
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Judge label {model!r} does not exist under {judge_dir}. {guidance}")
    chunks = sorted((model_dir / "inputs").glob("chunk_*.json"))
    if not chunks:
        raise ValueError(f"No input chunks for judge label {model!r} under {judge_dir}. {guidance}")
    return chunks


def expected_recs(model: str, judge_dir: Path = JUDGE) -> Dict[str, int]:
    """task_id -> number of recommendations expected (from input chunks)."""
    exp: Dict[str, int] = {}
    for f in _input_chunks(model, judge_dir):
        for p in json.load(open(f, encoding="utf-8")):
            exp[p["task_id"]] = len(p["recommendations"])
    return exp


def n_constraints_map(model: str, judge_dir: Path = JUDGE) -> Dict[str, int]:
    m: Dict[str, int] = {}
    for f in _input_chunks(model, judge_dir):
        for p in json.load(open(f, encoding="utf-8")):
            m[p["task_id"]] = p["n_constraints"]
    return m


def summarize_model(model: str, judge_dir: Path = JUDGE) -> Dict[str, Any]:
    verdict_dir = judge_dir / model / "verdicts"
    rows = []
    for f in sorted(verdict_dir.glob("chunk_*.jsonl")):
        rows.extend(read_jsonl(f))

    exp = expected_recs(model, judge_dir)
    nc = n_constraints_map(model, judge_dir)

    # completeness
    got_per_task: Counter = Counter(r["task_id"] for r in rows)
    expected_total = sum(v for v in exp.values())
    missing_tasks = [t for t, n in exp.items() if got_per_task.get(t, 0) != n and n > 0]

    verdicts = Counter(r["verdict"] for r in rows)
    total = sum(verdicts.values())

    # per-query aggregation
    by_task: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_task[r["task_id"]].append(r)

    ratios = []
    success_all = 0
    nonempty = 0
    for tid, n in exp.items():
        if n == 0:
            continue
        nonempty += 1
        vr = [x["verdict"] for x in by_task.get(tid, [])]
        q = sum(v == "qualified" for v in vr)
        if vr:
            ratios.append(q / len(vr))
        success_all += int(q > 0)

    # constraint satisfaction (constraints_met)
    cm_sum = 0
    cm_den = 0
    acc_cm = []
    unacc_cm = []
    for r in rows:
        tid = r["task_id"]
        k = nc.get(tid, 0)
        if k <= 0:
            continue
        met = r.get("constraints_met")
        if isinstance(met, int):
            cm_sum += met
            cm_den += k
        if r["verdict"] == "acceptable":
            acc_cm.append(met)
        elif r["verdict"] == "unacceptable":
            unacc_cm.append(met)

    def avg(xs):
        xs = [x for x in xs if isinstance(x, int)]
        return round(sum(xs) / len(xs), 4) if xs else None

    result = {
        "model": model,
        "label": MODEL_LABELS.get(model, model),
        "expected_recommendations": expected_total,
        "judged_recommendations": total,
        "nonempty_queries": nonempty,
        "incomplete_tasks": len(missing_tasks),
        "incomplete_task_ids": missing_tasks[:20],
        "verdict_counts": dict(verdicts),
        "qualified_pois_per_query": round(verdicts["qualified"] / nonempty, 4) if nonempty else None,
        "macro_qualified_ratio": round(sum(ratios) / len(ratios), 4) if ratios else None,
        "micro_qualified_ratio": round(verdicts["qualified"] / total, 4) if total else None,
        "success_at_k": round(success_all / nonempty, 4) if nonempty else None,
        "constraint_satisfaction_rate": round(cm_sum / cm_den, 4) if cm_den else None,
        "avg_constraints_met_acceptable": avg(acc_cm),
        "avg_constraints_met_unacceptable": avg(unacc_cm),
    }
    (judge_dir / model / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge-dir", type=Path, default=JUDGE)
    parser.add_argument("models", nargs="*", help="Judge labels (including custom labels)")
    args = parser.parse_intermixed_args(argv)
    judge_dir = args.judge_dir.resolve()
    only = args.models or [
        model for model in MODEL_LABELS
        if any((judge_dir / model / "inputs").glob("chunk_*.json"))
    ]
    if not only:
        parser.error(
            f"No judge inputs available under {judge_dir}. "
            "Run build_qoder_judge_inputs.py and judge the input chunks first."
        )
    for model in only:
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", model) is None:
            parser.error(f"Invalid judge label {model!r}: use one safe path component")
        try:
            _input_chunks(model, judge_dir)
        except (FileNotFoundError, ValueError) as exc:
            parser.error(str(exc))

    results = []
    for model in only:
        r = summarize_model(model, judge_dir)
        results.append(r)
        print(json.dumps({k: r[k] for k in (
            "label", "judged_recommendations", "expected_recommendations",
            "incomplete_tasks", "verdict_counts", "qualified_pois_per_query",
            "micro_qualified_ratio", "success_at_k", "constraint_satisfaction_rate")},
            ensure_ascii=False), flush=True)

    # overall markdown table
    results.sort(key=lambda x: (x["micro_qualified_ratio"] or 0), reverse=True)
    lines = [
        "| Model | Recs judged | Qualified/query | Micro Qual | Macro Qual | Success@k | Constraint-sat | Incomplete |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            "| {label} | {jr}/{er} | {qpq} | {mi} | {ma} | {sk} | {cs} | {inc} |".format(
                label=r["label"], jr=r["judged_recommendations"], er=r["expected_recommendations"],
                qpq=r["qualified_pois_per_query"], mi=r["micro_qualified_ratio"],
                ma=r["macro_qualified_ratio"], sk=r["success_at_k"],
                cs=r["constraint_satisfaction_rate"], inc=r["incomplete_tasks"])
        )
    (judge_dir / "_overall.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (judge_dir / "_overall.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nWrote {judge_dir / '_overall.md'} and {judge_dir / '_overall.json'}", flush=True)


if __name__ == "__main__":
    main()
