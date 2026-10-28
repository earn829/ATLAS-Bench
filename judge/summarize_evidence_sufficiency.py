"""Summarize the DeepSeek-Pro evidence-sufficiency judge (0-5) over all 14 models.

Per model (deduped by task_id): mean evidence_score, 0-5 distribution, low% (0-1),
%hard-deficient (>=1 hard constraint not sufficient), %comparison_single_candidate,
%fabricated_or_wrong_object. Writes judge/evidence_sufficiency_leaderboard.{md,json}.
"""

from __future__ import annotations

import glob
import json
import os
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JDIR = ROOT / "judge" / "evidence_sufficiency_deepseek_pro"
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
                if line.strip():
                    r = json.loads(line)
                    if r.get("status") == "completed":
                        best[r["task_id"]] = r["judgment"]
        if not best:
            continue
        js = list(best.values())
        sc = [j["evidence_score"] for j in js]
        d = Counter(sc)
        n = len(sc)
        rows.append({
            "label": label, "model": PRETTY.get(label, label), "judged": n,
            "mean": round(sum(sc) / n, 4),
            "dist": {str(k): d[k] for k in range(0, 6)},
            "low_pct": round(100 * (d[0] + d[1]) / n, 1),
            "hard_def_pct": round(100 * sum(1 for j in js if j.get("hard_deficient", 0) >= 1) / n, 1),
            "comp_single_pct": round(100 * sum(1 for j in js if j.get("comparison_single_candidate")) / n, 1),
            "fabricated_pct": round(100 * sum(1 for j in js if j.get("fabricated_or_wrong_object")) / n, 1),
        })
    rows.sort(key=lambda r: -r["mean"])
    (JDIR.parent / "evidence_sufficiency_leaderboard.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    L = ["# Evidence sufficiency (DeepSeek-Pro judge, anti-inflation caps)", "",
         "hard-def% = 有刚性约束证据非sufficient的query占比; comp-single% = 比较任务只核验单候选; fabric% = 核心证据错接/过期/编造。", "",
         "| # | 模型 | judged | mean | low%(0-1) | hard-def% | comp-single% | fabric% | dist 0/1/2/3/4/5 |",
         "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for i, r in enumerate(rows, 1):
        dd = r["dist"]
        L.append("| %d | %s | %d | %.3f | %.1f%% | %.1f%% | %.1f%% | %.1f%% | %d/%d/%d/%d/%d/%d |" % (
            i, r["model"], r["judged"], r["mean"], r["low_pct"], r["hard_def_pct"],
            r["comp_single_pct"], r["fabricated_pct"],
            dd["0"], dd["1"], dd["2"], dd["3"], dd["4"], dd["5"]))
    md = "\n".join(L)
    (JDIR.parent / "evidence_sufficiency_leaderboard.md").write_text(md + "\n", encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
