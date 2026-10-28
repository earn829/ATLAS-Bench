"""Summarize the DeepSeek-Pro tool-selection judge over all 14 models.

Per model (deduped by task_id): mean tool_score, 1-5 distribution, low% (1-2),
material% (tool_errors or unverified), unverified%, verified_but_negative% (rule C
exemption in action: verified but came back negative/empty, NOT penalized),
param-issue% and redundant%. Writes judge/tool_selection_leaderboard.{md,json}.
"""

from __future__ import annotations

import glob
import json
import os
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JDIR = ROOT / "judge" / "tool_selection_deepseek_pro"
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
        sc = [j["tool_score"] for j in js]
        d = Counter(sc)
        n = len(sc)
        mat = sum(1 for j in js if j.get("tool_errors") or j.get("unverified"))
        rows.append({
            "label": label, "model": PRETTY.get(label, label), "judged": n,
            "mean": round(sum(sc) / n, 4),
            "dist": {str(k): d[k] for k in range(1, 6)},
            "low_pct": round(100 * (d[1] + d[2]) / n, 1),
            "material_pct": round(100 * mat / n, 1),
            "unverified_pct": round(100 * sum(1 for j in js if j.get("unverified")) / n, 1),
            "vbn_pct": round(100 * sum(1 for j in js if j.get("verified_but_negative")) / n, 1),
            "param_pct": round(100 * sum(1 for j in js if j.get("param_issues")) / n, 1),
            "redundant_pct": round(100 * sum(1 for j in js if j.get("redundant_calls")) / n, 1),
        })
    rows.sort(key=lambda r: -r["mean"])
    (JDIR.parent / "tool_selection_leaderboard.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    L = ["# Tool selection (DeepSeek-Pro judge, anti-inflation caps)", "",
         "material% = 有 tool_errors 或 unverified(实质失误); vbn% = 已正确核验但结果否定/空(规则C豁免,不扣分)。", "",
         "| # | 模型 | judged | mean | low%(1-2) | material% | unverified% | vbn%(豁免) | param% | redundant% | dist 1/2/3/4/5 |",
         "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for i, r in enumerate(rows, 1):
        dd = r["dist"]
        L.append("| %d | %s | %d | %.3f | %.1f%% | %.1f%% | %.1f%% | %.1f%% | %.1f%% | %.1f%% | %d/%d/%d/%d/%d |" % (
            i, r["model"], r["judged"], r["mean"], r["low_pct"], r["material_pct"],
            r["unverified_pct"], r["vbn_pct"], r["param_pct"], r["redundant_pct"],
            dd["1"], dd["2"], dd["3"], dd["4"], dd["5"]))
    md = "\n".join(L)
    (JDIR.parent / "tool_selection_leaderboard.md").write_text(md + "\n", encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
