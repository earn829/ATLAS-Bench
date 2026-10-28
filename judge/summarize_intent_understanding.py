"""Summarize the DeepSeek-Pro intent-understanding judge over all 14 models.

Per model (deduped by task_id): mean intent_score, the 1-5 distribution, low% (1-2),
material-failure% (score<=2, i.e. a cited hard-constraint miss or intent-drift event),
hard-clean% (no hard_missed), and perfect% (score 5). Writes
judge/intent_understanding_leaderboard.{md,json}.
"""

from __future__ import annotations

import glob
import json
import os
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JDIR = ROOT / "judge" / "intent_understanding_deepseek_pro"
PRETTY = {
    "qwen35_4b_nothink": "Qwen3.5-4B w/o", "qwen35_4b_think": "Qwen3.5-4B w/",
    "qwen36_a3b_nothink": "Qwen3.6-A3B w/o", "qwen36_a3b_think": "Qwen3.6-A3B w/",
    "qwen38_27b_nothink": "Qwen3.8-27B w/o", "qwen38_27b_think": "Qwen3.8-27B w/",
    "qwen38_max": "Qwen3.8-Max", "deepseek_v41_flash": "DeepSeek-V4.1-Flash",
    "claude5_sonnet": "Claude-5-Sonnet", "gpt56_luna": "GPT-5.6-Luna",
    "gpt6_luna": "GPT-6-Luna", "glm53": "GLM-5.3", "kimi_k3": "Kimi-K3",
    "gemini38_flash": "Gemini-3.8-Flash",
}


def main() -> None:
    rows = []
    for path in sorted(glob.glob(str(JDIR / "*.jsonl"))):
        label = os.path.basename(path)[: -len(".jsonl")]
        best = {}
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("status") == "completed":
                    best[r["task_id"]] = r["judgment"]
        if not best:
            continue
        js = list(best.values())
        sc = [j["intent_score"] for j in js]
        d = Counter(sc)
        n = len(sc)
        hard_clean = sum(1 for j in js if not j.get("hard_missed"))
        rows.append({
            "label": label, "model": PRETTY.get(label, label), "judged": n,
            "mean": round(sum(sc) / n, 4),
            "dist": {str(k): d[k] for k in range(1, 6)},
            "low_pct": round(100 * (d[1] + d[2]) / n, 1),
            "material_pct": round(100 * (d[1] + d[2]) / n, 1),
            "hard_clean_pct": round(100 * hard_clean / n, 1),
            "perfect_pct": round(100 * d[5] / n, 1),
        })
    rows.sort(key=lambda r: -r["mean"])
    (JDIR.parent / "intent_understanding_leaderboard.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    L = ["# Intent understanding (DeepSeek-Pro judge, anti-inflation caps)", "",
         "mean=均分; low%/material%=打1-2分(有实质意图失误)占比; hard-clean%=无刚性约束遗漏; perfect%=打5分占比。", "",
         "| # | 模型 | judged | mean | low%(1-2) | hard-clean% | perfect%(5) | dist 1/2/3/4/5 |",
         "|---|---|---:|---:|---:|---:|---:|---|"]
    for i, r in enumerate(rows, 1):
        dd = r["dist"]
        L.append("| %d | %s | %d | %.3f | %.1f%% | %.1f%% | %.1f%% | %d/%d/%d/%d/%d |" % (
            i, r["model"], r["judged"], r["mean"], r["low_pct"], r["hard_clean_pct"],
            r["perfect_pct"], dd["1"], dd["2"], dd["3"], dd["4"], dd["5"]))
    md = "\n".join(L)
    (JDIR.parent / "intent_understanding_leaderboard.md").write_text(md + "\n", encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
