"""Combine the programmatic answer-pass judge with the DeepSeek-Pro factual judge.

Per query, combined_correct = programmatic_passed AND factual_correct. Rationale: a
failed programmatic answer (inconsistent no_exact_match, fabricated evidence id, or
evidence owned by another POI) is itself an unfaithful/incorrect answer, so it is
forced to factually wrong even when the LLM factual judge returned factual_correct=1.

Joins outputs/<rundir>/subset_1200.final.jsonl (programmatic_judge.passed) with
judge/factual_accuracy_deepseek_pro/<label>.jsonl (judgment.factual_correct) on task_id.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parent.parent

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
    "qwen35_4b_nothink": "Qwen3.5-4B w/o", "qwen35_4b_think": "Qwen3.5-4B w/",
    "qwen36_a3b_nothink": "Qwen3.6-A3B w/o", "qwen36_a3b_think": "Qwen3.6-A3B w/",
    "qwen38_27b_nothink": "Qwen3.8-27B w/o", "qwen38_27b_think": "Qwen3.8-27B w/",
    "qwen38_max": "Qwen3.8-Max", "deepseek_v41_flash": "DeepSeek-V4.1-Flash",
    "claude5_sonnet": "Claude-5-Sonnet", "gpt56_luna": "GPT-5.6-Luna",
    "gpt6_luna": "GPT-6-Luna", "glm53": "GLM-5.3", "kimi_k3": "Kimi-K3",
    "gemini38_flash": "Gemini-3.8-Flash",
}
JUDGE_DIR = ROOT / "judge" / "factual_accuracy_deepseek_pro"


def load_pass(rundir: str) -> Dict[str, bool]:
    path = ROOT / "outputs" / rundir / "subset_1200.final.jsonl"
    out: Dict[str, bool] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            tid = rec.get("task_id")
            pj = rec.get("programmatic_judge")
            out[tid] = bool(isinstance(pj, dict) and pj.get("passed"))
    return out


def load_factual(label: str) -> Dict[str, int]:
    path = JUDGE_DIR / f"{label}.jsonl"
    out: Dict[str, int] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("status") != "completed":
                continue
            j = rec.get("judgment") or {}
            fc = j.get("factual_correct")
            if fc in (0, 1):
                out[rec.get("task_id")] = int(fc)
    return out


def main() -> None:
    rows = []
    for label, rundir in LABEL2RUN.items():
        passed = load_pass(rundir)
        factual = load_factual(label)
        tids = sorted(set(passed) | set(factual))
        n = len(tids)
        n_pass = n_fc = n_comb = flip = 0
        for tid in tids:
            p = passed.get(tid, False)
            f = factual.get(tid, 0)
            n_pass += int(p)
            n_fc += int(f)
            comb = int(p and f)
            n_comb += comb
            if f == 1 and not p:
                flip += 1
        rows.append(
            {
                "label": label,
                "model": PRETTY[label],
                "n": n,
                "answer_pass_rate": 100 * n_pass / n if n else 0,
                "factual_acc": 100 * n_fc / n if n else 0,
                "combined_acc": 100 * n_comb / n if n else 0,
                "flip_fc1_pass0": flip,
            }
        )

    rows.sort(key=lambda r: -r["combined_acc"])
    (ROOT / "judge" / "combined_passrate_factual.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    hdr = "| # | Model | N | AnswerPass% | Factual% | Combined% | flip(fc=1&pass=0) |"
    sep = "|---|---|---:|---:|---:|---:|---:|"
    lines = [hdr, sep]
    for i, r in enumerate(rows, 1):
        lines.append(
            "| %d | %s | %d | %.2f | %.2f | **%.2f** | %d |"
            % (i, r["model"], r["n"], r["answer_pass_rate"], r["factual_acc"],
               r["combined_acc"], r["flip_fc1_pass0"])
        )
    table = "\n".join(lines)
    (ROOT / "judge" / "combined_passrate_factual.md").write_text(table + "\n", encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
