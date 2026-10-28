"""Summarize the DeepSeek-Pro tool-use-flexibility judge over all 14 models.

Per model reports: judged/failed, the 1-5 score distribution, the mean over all
judged queries, and the CONDITIONAL mean over only the queries where adaptation was
actually needed (adaptation_needed=true) — the subset where flexibility is exercised
and discrimination lives (analogous to conditional ILD for diversity).

Reads judge/tool_flexibility_deepseek_pro/<label>.jsonl; writes
judge/tool_flexibility_leaderboard.{md,json}.
"""

from __future__ import annotations

import glob
import json
import os
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JDIR = ROOT / "judge" / "tool_flexibility_deepseek_pro"
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
        scores = []
        cond = []
        failed = 0
        dist = Counter()
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("status") != "completed":
                    failed += 1
                    continue
                j = r.get("judgment") or {}
                s = j.get("flexibility_score")
                if not isinstance(s, int):
                    continue
                scores.append(s)
                dist[s] += 1
                if j.get("adaptation_needed") is True:
                    cond.append(s)
        if not scores:
            continue
        rows.append(
            {
                "label": label,
                "model": PRETTY.get(label, label),
                "judged": len(scores),
                "failed": failed,
                "mean_all": round(sum(scores) / len(scores), 4),
                "n_adaptation_needed": len(cond),
                "pct_adaptation_needed": round(100 * len(cond) / len(scores), 1),
                "mean_conditional": round(sum(cond) / len(cond), 4) if cond else None,
                "dist": {str(k): dist[k] for k in range(1, 6)},
                "low_share": round(100 * (dist[1] + dist[2]) / len(scores), 1),
            }
        )

    rows.sort(key=lambda r: -r["mean_all"])
    (JDIR.parent / "tool_flexibility_leaderboard.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    L = ["# Strategy adaptation (DeepSeek-Pro judge, anti-inflation caps)", ""]
    L.append("mean_all = 全部已判 query 均分;mean_cond = 仅在 adaptation_needed=true 的子集均分(策略调整真正被考验处);low% = 打 1-2 分占比。")
    L.append("")
    L.append("| # | 模型 | judged | mean_all | mean_cond | adapt-needed% | low%(1-2) | dist 1/2/3/4/5 |")
    L.append("|---|---|---:|---:|---:|---:|---:|---|")
    for i, r in enumerate(rows, 1):
        d = r["dist"]
        L.append("| %d | %s | %d | %.3f | %s | %.1f%% | %.1f%% | %d/%d/%d/%d/%d |" % (
            i, r["model"], r["judged"], r["mean_all"],
            ("%.3f" % r["mean_conditional"]) if r["mean_conditional"] is not None else "--",
            r["pct_adaptation_needed"], r["low_share"],
            d["1"], d["2"], d["3"], d["4"], d["5"]))
    md = "\n".join(L)
    (JDIR.parent / "tool_flexibility_leaderboard.md").write_text(md + "\n", encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
