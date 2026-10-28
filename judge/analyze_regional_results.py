"""Regional analysis of existing ATLAS-Bench results; no new model/judge calls.

Aggregates the 14 finalized runs, Qoder recommendation verdicts, DeepSeek-Pro evidence
and factual judges by Chaoyang, Huangpu, and Dabancheng. Reports raw city metrics,
within-city model rankings, cross-city directional consistency, and descriptive adjusted
city effects controlling for model, task type, and constraint count via OLS fixed effects.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
TASK_PATH = ROOT / "query" / "subset_1200.json"
OUT_DIR = ROOT / "outputs" / "regional_analysis"
GOOD = {"qualified", "acceptable"}

LABEL2RUN = {
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
PRETTY = {
    "qwen35_4b_nothink": "Qwen-4B−", "qwen35_4b_think": "Qwen-4B+",
    "qwen36_a3b_nothink": "Qwen-A3B−", "qwen36_a3b_think": "Qwen-A3B+",
    "qwen38_27b_nothink": "Qwen-27B−", "qwen38_27b_think": "Qwen-27B+",
    "qwen38_max": "Qwen-Max", "deepseek_v41_flash": "DeepSeek",
    "claude5_sonnet": "Claude", "gpt56_luna": "GPT-5.6",
    "gpt6_luna": "GPT-6", "glm53": "GLM", "kimi_k3": "Kimi",
    "gemini38_flash": "Gemini",
}
CITY_SHORT = {
    "北京·朝阳区": "Chaoyang", "上海·黄浦区": "Huangpu",
    "乌鲁木齐·达坂城区": "Dabancheng",
}
CITY_ORDER = ["Chaoyang", "Huangpu", "Dabancheng"]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_tasks() -> dict[str, dict[str, Any]]:
    out = {}
    for item in json.load(TASK_PATH.open(encoding="utf-8")):
        record = item["record"]
        constraints = record.get("constraints") or []
        out[record["id"]] = {
            "task_id": record["id"], "city": CITY_SHORT[item["city"]],
            "city_full": item["city"], "type": item["type"], "bucket": item["bucket"],
            "n_constraints": len(constraints),
            "n_hard": sum(c.get("h") == "刚性" for c in constraints),
            "n_soft": sum(c.get("h") == "柔性" for c in constraints),
            "satisfiable": (record.get("answer") or {}).get("satisfiable"),
        }
    return out


def load_verdicts(label: str) -> dict[str, list[dict[str, Any]]]:
    by_task = defaultdict(list)
    for path in sorted((ROOT / "judge" / label / "verdicts").glob("chunk_*.jsonl")):
        for row in load_jsonl(path):
            by_task[row["task_id"]].append(row)
    for rows in by_task.values():
        rows.sort(key=lambda row: int(row["rank"]))
    return by_task


def load_judge(path: Path, score_key: str) -> dict[str, float]:
    out = {}
    for row in load_jsonl(path):
        if row.get("status") != "completed":
            continue
        judgment = row.get("judgment") or {}
        value = judgment.get(score_key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out[row["task_id"]] = float(value)
    return out


def token_count(record: dict[str, Any]) -> int:
    total = 0
    for usage in (record.get("metrics") or {}).get("usage_events") or []:
        if isinstance(usage, dict):
            total += int(usage.get("prompt_tokens") or 0) + int(usage.get("completion_tokens") or 0)
    return total


def query_rows(tasks: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for label, rundir in LABEL2RUN.items():
        records = {row["task_id"]: row for row in load_jsonl(
            ROOT / "outputs" / rundir / "subset_1200.final.jsonl")}
        verdicts = load_verdicts(label)
        evidence = load_judge(ROOT / "judge" / "evidence_sufficiency_deepseek_pro" / f"{label}.jsonl", "evidence_score")
        factual = load_judge(ROOT / "judge" / "factual_accuracy_deepseek_pro" / f"{label}.jsonl", "factual_correct")
        for task_id, task in tasks.items():
            record = records[task_id]
            answer = record.get("answer") or {}
            recs = answer.get("recommendations") or []
            vrows = verdicts.get(task_id, [])
            n_constraints = task["n_constraints"]
            den = len(vrows) * n_constraints
            cr_num = sum(int(v.get("constraints_met") or 0) for v in vrows)
            csr_num = sum(int(v.get("constraints_met") or 0) for v in vrows if v.get("verdict") in GOOD)
            metrics = record.get("metrics") or {}
            programmatic = bool((record.get("programmatic_judge") or {}).get("passed"))
            factual_correct = bool(factual.get(task_id, 0))
            no_exact = bool(answer.get("no_exact_match"))
            rows.append({
                **task, "label": label, "model": PRETTY[label],
                "n_recs": len(recs),
                "success": int(any(v.get("verdict") in GOOD for v in vrows)),
                "csr_num": csr_num, "cr_num": cr_num, "constraint_den": den,
                "csr_query": csr_num / den if den else None,
                "cr_query": cr_num / den if den else None,
                "combined_factual": int(programmatic and factual_correct),
                "programmatic_pass": int(programmatic), "factual_correct": int(factual_correct),
                "evidence_score": evidence.get(task_id),
                "tool_calls": float(metrics.get("tool_calls") or 0),
                "tool_rounds": float(metrics.get("tool_rounds") or 0),
                "tokens": float(token_count(record)),
                "latency": float(metrics.get("elapsed_seconds") or 0),
                "no_exact_match": int(no_exact),
                "correct_unsat_refusal": int(len(recs) == 0 and no_exact),
            })
    return rows


def composition(tasks: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for city in CITY_ORDER:
        rows = [row for row in tasks.values() if row["city"] == city]
        sat = Counter(row["satisfiable"] for row in rows)
        out.append({
            "city": city, "queries": len(rows), "should_answer": len(rows) - sat[False],
            "satisfiable_true": sat[True], "unsatisfiable": sat[False], "unannotated": sat[None],
            "avg_constraints": statistics.mean(row["n_constraints"] for row in rows),
            "avg_hard": statistics.mean(row["n_hard"] for row in rows),
            "avg_soft": statistics.mean(row["n_soft"] for row in rows),
            "single": sum(row["bucket"] == "single" for row in rows),
            "multi": sum(row["bucket"] == "multi" for row in rows),
        })
    return out


def aggregate_model_city(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["label"], row["city"])].append(row)
    out = []
    for (label, city), values in grouped.items():
        should = [row for row in values if row["satisfiable"] is not False]
        unsat = [row for row in values if row["satisfiable"] is False]
        den = sum(row["constraint_den"] for row in values)
        evidence = [row["evidence_score"] for row in values if row["evidence_score"] is not None]
        out.append({
            "label": label, "model": PRETTY[label], "city": city,
            "queries": len(values), "should_queries": len(should), "unsat_queries": len(unsat),
            "csr": sum(row["csr_num"] for row in values) / den if den else None,
            "cr": sum(row["cr_num"] for row in values) / den if den else None,
            "success": statistics.mean(row["success"] for row in should),
            "recs_per_query": statistics.mean(row["n_recs"] for row in should),
            "factual_accuracy": statistics.mean(row["combined_factual"] for row in values),
            "evidence_sufficiency": statistics.mean(evidence) if evidence else None,
            "evidence_judged": len(evidence),
            "tool_calls": statistics.mean(row["tool_calls"] for row in values),
            "tool_rounds": statistics.mean(row["tool_rounds"] for row in values),
            "tokens": statistics.mean(row["tokens"] for row in values),
            "latency_p50": statistics.median(row["latency"] for row in values),
            "unsat_refusal": statistics.mean(row["correct_unsat_refusal"] for row in unsat),
        })
    return out


def macro_city(model_city: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = ["csr", "cr", "success", "recs_per_query", "factual_accuracy",
               "evidence_sufficiency", "tool_calls", "tool_rounds", "tokens",
               "latency_p50", "unsat_refusal"]
    out = []
    for city in CITY_ORDER:
        values = [row for row in model_city if row["city"] == city]
        row = {"city": city, "models": len(values)}
        for metric in metrics:
            xs = [value[metric] for value in values if value[metric] is not None]
            row[metric] = statistics.mean(xs)
            row[metric + "_sd"] = statistics.pstdev(xs)
        out.append(row)
    return out


def adjusted_effects(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Descriptive OLS with model, task type, and constraint-count fixed effects.
    specs = {
        "csr": (lambda r: r["csr_query"] is not None),
        "cr": (lambda r: r["cr_query"] is not None),
        "success": (lambda r: r["satisfiable"] is not False),
        "recs_per_query": (lambda r: r["satisfiable"] is not False),
        "factual_accuracy": (lambda r: True),
        "evidence_sufficiency": (lambda r: r["evidence_score"] is not None),
        "tool_calls": (lambda r: True),
        "tool_rounds": (lambda r: True),
        "tokens": (lambda r: True),
        "unsat_refusal": (lambda r: r["satisfiable"] is False),
    }
    ykey = {
        "csr": "csr_query", "cr": "cr_query",
        "success": "success", "recs_per_query": "n_recs",
        "factual_accuracy": "combined_factual", "evidence_sufficiency": "evidence_score",
        "tool_calls": "tool_calls", "tool_rounds": "tool_rounds", "tokens": "tokens",
        "unsat_refusal": "correct_unsat_refusal",
    }
    model_levels = list(LABEL2RUN)
    type_levels = sorted({row["type"] for row in rows})
    out = []
    for metric, keep in specs.items():
        sample = [row for row in rows if keep(row)]
        columns = [np.ones(len(sample))]
        names = ["intercept"]
        for city in ("Huangpu", "Dabancheng"):
            columns.append(np.array([int(row["city"] == city) for row in sample], dtype=float))
            names.append(city + "_vs_Chaoyang")
        for level in model_levels[1:]:
            columns.append(np.array([int(row["label"] == level) for row in sample], dtype=float))
            names.append("model=" + level)
        for level in type_levels[1:]:
            columns.append(np.array([int(row["type"] == level) for row in sample], dtype=float))
            names.append("type=" + level)
        columns.append(np.array([row["n_constraints"] for row in sample], dtype=float))
        names.append("n_constraints")
        columns.append(np.array([row["n_soft"] for row in sample], dtype=float))
        names.append("n_soft")
        x = np.column_stack(columns)
        y = np.array([row[ykey[metric]] for row in sample], dtype=float)
        beta, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
        for city_name in ("Huangpu_vs_Chaoyang", "Dabancheng_vs_Chaoyang"):
            idx = names.index(city_name)
            out.append({"metric": metric, "contrast": city_name, "effect": float(beta[idx]),
                        "n": len(sample), "controls": "model + type + n_constraints + n_soft"})
    return out


def direction_counts(model_city: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_model = defaultdict(dict)
    for row in model_city:
        by_model[row["label"]][row["city"]] = row
    metrics = ["success", "recs_per_query", "evidence_sufficiency", "tool_calls",
               "tokens", "unsat_refusal", "factual_accuracy", "csr", "cr"]
    out = []
    for metric in metrics:
        for a, b in (("Huangpu", "Chaoyang"), ("Dabancheng", "Chaoyang"),
                     ("Dabancheng", "Huangpu")):
            deltas = [cities[a][metric] - cities[b][metric] for cities in by_model.values()]
            out.append({"metric": metric, "contrast": f"{a}-{b}",
                        "positive_models": sum(delta > 0 for delta in deltas),
                        "negative_models": sum(delta < 0 for delta in deltas),
                        "mean_delta": statistics.mean(deltas),
                        "median_delta": statistics.median(deltas)})
    return out


def rank_correlations(model_city: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_city = defaultdict(dict)
    for row in model_city:
        by_city[row["city"]][row["label"]] = row
    metrics = ["success", "csr", "cr", "evidence_sufficiency", "factual_accuracy",
               "tool_calls", "tokens"]
    out = []
    labels = list(LABEL2RUN)
    for metric in metrics:
        for city_a, city_b in (("Chaoyang", "Huangpu"), ("Chaoyang", "Dabancheng"),
                               ("Huangpu", "Dabancheng")):
            a = [by_city[city_a][label][metric] for label in labels]
            b = [by_city[city_b][label][metric] for label in labels]
            rho, p = spearmanr(a, b)
            out.append({"metric": metric, "city_a": city_a, "city_b": city_b,
                        "spearman_rho": float(rho), "p_value": float(p)})
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def pct(x: float) -> str:
    return f"{100*x:.1f}%"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks()
    rows = query_rows(tasks)
    comp = composition(tasks)
    model_city = aggregate_model_city(rows)
    macro = macro_city(model_city)
    adjusted = adjusted_effects(rows)
    directions = direction_counts(model_city)
    rank_corr = rank_correlations(model_city)

    write_csv(OUT_DIR / "query_composition.csv", comp)
    write_csv(OUT_DIR / "model_city_metrics.csv", sorted(model_city, key=lambda r: (r["city"], r["model"])))
    write_csv(OUT_DIR / "city_macro_metrics.csv", macro)
    write_csv(OUT_DIR / "adjusted_city_effects.csv", adjusted)
    write_csv(OUT_DIR / "direction_counts.csv", directions)
    write_csv(OUT_DIR / "city_rank_correlations.csv", rank_corr)
    (OUT_DIR / "query_level_metrics.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

    lines = ["# Existing-result regional analysis", "",
             "No model or judge was rerun. All values are re-aggregated from existing finalized trajectories and judge files.", "",
             "## Query composition", "",
             "| Region | Queries | Should-answer | Unsat | Avg constraints | Avg hard | Avg soft |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in comp:
        lines.append(f"| {row['city']} | {row['queries']} | {row['should_answer']} | {row['unsatisfiable']} | {row['avg_constraints']:.2f} | {row['avg_hard']:.2f} | {row['avg_soft']:.2f} |")
    lines += ["", "## Macro-average across 14 models (raw, unadjusted)", "",
              "| Region | CSR | CR | Success@k | Recs/q | Factual | Evidence | Tool calls | Rounds | Tokens | P50 s | Unsat refusal |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in macro:
        lines.append(f"| {row['city']} | {pct(row['csr'])} | {pct(row['cr'])} | {pct(row['success'])} | {row['recs_per_query']:.2f} | {pct(row['factual_accuracy'])} | {row['evidence_sufficiency']:.2f} | {row['tool_calls']:.2f} | {row['tool_rounds']:.2f} | {row['tokens']/1000:.1f}K | {row['latency_p50']:.1f} | {pct(row['unsat_refusal'])} |")

    lines += ["", "## Within-region top models", ""]
    for city in CITY_ORDER:
        values = [row for row in model_city if row["city"] == city]
        lines.append(f"### {city}")
        for metric in ("success", "csr", "evidence_sufficiency", "factual_accuracy"):
            top = sorted(values, key=lambda row: row[metric] or -1, reverse=True)[:3]
            value_text = ", ".join(f"{row['model']}={100*row[metric]:.1f}%" if metric != "evidence_sufficiency" else f"{row['model']}={row[metric]:.2f}" for row in top)
            lines.append(f"- {metric}: {value_text}")
        lines.append("")

    lines += ["## Adjusted region contrasts", "",
              "Controls: model fixed effects, task type, number of constraints, and number of soft constraints. Effects remain descriptive, not causal.", "",
              "| Metric | Huangpu − Chaoyang | Dabancheng − Chaoyang |",
              "|---|---:|---:|"]
    by_metric = defaultdict(dict)
    for row in adjusted:
        by_metric[row["metric"]][row["contrast"]] = row["effect"]
    percentage_metrics = {"csr", "cr", "success", "factual_accuracy", "unsat_refusal"}
    for metric in by_metric:
        a = by_metric[metric]["Huangpu_vs_Chaoyang"]
        b = by_metric[metric]["Dabancheng_vs_Chaoyang"]
        if metric in percentage_metrics:
            lines.append(f"| {metric} | {100*a:+.1f} pp | {100*b:+.1f} pp |")
        elif metric == "tokens":
            lines.append(f"| {metric} | {a/1000:+.1f}K | {b/1000:+.1f}K |")
        else:
            lines.append(f"| {metric} | {a:+.2f} | {b:+.2f} |")

    lines += ["", "## Cross-city model-rank stability (Spearman rho)", "",
              "| Metric | Chaoyang–Huangpu | Chaoyang–Dabancheng | Huangpu–Dabancheng |",
              "|---|---:|---:|---:|"]
    corr_map = {(r["metric"], r["city_a"], r["city_b"]): r["spearman_rho"] for r in rank_corr}
    for metric in ("success", "csr", "cr", "evidence_sufficiency", "factual_accuracy", "tool_calls", "tokens"):
        lines.append(f"| {metric} | {corr_map[(metric, 'Chaoyang', 'Huangpu')]:.2f} | {corr_map[(metric, 'Chaoyang', 'Dabancheng')]:.2f} | {corr_map[(metric, 'Huangpu', 'Dabancheng')]:.2f} |")

    md = "\n".join(lines) + "\n"
    (OUT_DIR / "REGIONAL_ANALYSIS.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
