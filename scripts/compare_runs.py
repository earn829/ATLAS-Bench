"""Render a cross-run comparison table from finalized run files.

Every metric is produced by finalize_run.summarize, so this script cannot drift from the
per-run summaries -- there is a single definition of each number.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from finalize_run import dedupe_records, read_jsonl, summarize

# (表格行名, 从summary取值的函数, 是否为"越大越好")
ROWS: List[tuple] = [
    ("任务数", lambda s: s["records"], None),
    ("完成/失败", lambda s: "%d / %d" % (s["completed"], s["failed"]), None),
    ("失败类型", lambda s: ", ".join("%s×%d" % kv for kv in sorted(s["failure_types"].items())) or "—", None),
    ("答案整体通过率", lambda s: pct(s["programmatic_judge"]["answer_pass_rate"]), True),
    ("no_exact一致率", lambda s: pct(s["programmatic_judge"]["no_exact_match_consistency_rate"]), True),
    ("证据存在率", lambda s: pct(s["programmatic_judge"]["evidence_existence_accuracy"]), True),
    ("证据归属准确率", lambda s: pct(s["programmatic_judge"]["evidence_ownership_accuracy"]), True),
    ("归属错误数", lambda s: s["programmatic_judge"]["evidence_wrong_owner_count"], False),
    ("捏造引用数", lambda s: s["programmatic_judge"]["evidence_unknown_count"], False),
    ("证据引用总数", lambda s: s["programmatic_judge"]["evidence_reference_count"], None),
    ("推荐数均值", lambda s: num(s["recommendation_mean"]), None),
    ("零推荐条数", lambda s: s["recommendation_distribution"].get("0", 0), None),
    ("超过5条推荐的答案数", lambda s: s["over5_count"], None),
    ("平均tool_calls/条", lambda s: num(s["usage"]["tool_calls_avg"]), None),
    ("平均prompt_tokens/条", lambda s: num(s["usage"]["prompt_avg_per_task"], 0), None),
    ("平均completion_tokens/条", lambda s: num(s["usage"]["completion_avg_per_task"], 0), None),
    ("reasoning占completion", lambda s: reasoning_share(s), None),
    ("单次调用最大completion", lambda s: s["usage"]["max_completion_tokens_single_call"], None),
    ("单条耗时中位", lambda s: "%ss" % num(s["usage"]["elapsed_median_s"]), None),
    ("prompt tokens总计", lambda s: millions(s["usage"]["prompt_tokens_total"]), None),
    ("completion tokens总计", lambda s: millions(s["usage"]["completion_tokens_total"]), None),
]


def pct(value: Any) -> str:
    return "—" if value is None else "%.2f%%" % (100 * value)


def num(value: Any, digits: int = 2) -> str:
    return "—" if value is None else ("%.*f" % (digits, value))


def millions(value: Any) -> str:
    return "—" if not value else "%.2fM" % (value / 1e6)


def reasoning_share(summary: Dict[str, Any]) -> str:
    usage = summary["usage"]
    completion = usage.get("completion_tokens_total") or 0
    reasoning = usage.get("reasoning_tokens_total") or 0
    if not completion:
        return "—"
    return "%.1f%% (%s)" % (100 * reasoning / completion, millions(reasoning))


def load_summary(path: Path, tasks_order: List[str], config: Dict[str, Any]) -> Dict[str, Any]:
    records, attempts, parse_errors = dedupe_records(read_jsonl(path))
    if parse_errors:
        raise ValueError("%s 存在无法解析的行：%s" % (path, dict(parse_errors)))
    known = set(tasks_order)
    unexpected = sorted({r.get("task_id") for r in records} - known)
    if unexpected:
        raise ValueError("%s 含不在输入清单中的task：%s" % (path, unexpected[:5]))
    order = {task_id: index for index, task_id in enumerate(tasks_order)}
    records.sort(key=lambda r: order.get(r.get("task_id"), len(order)))
    return summarize(records, raw_line_count=len(records), attempts=attempts, config=config)


def markdown(labels: List[str], summaries: List[Dict[str, Any]], bold_best: bool) -> str:
    lines = [
        "| 指标 | " + " | ".join(labels) + " |",
        "|---|" + "---:|" * len(labels),
    ]
    for name, getter, better in ROWS:
        values = []
        for summary in summaries:
            try:
                values.append(getter(summary))
            except (KeyError, TypeError):
                values.append("—")
        values = [value if isinstance(value, str) else str(value) for value in values]
        if bold_best and better is not None:
            numbers = []
            for value in values:
                try:
                    numbers.append(float(value.rstrip("%").split(" ")[0]))
                except (ValueError, AttributeError):
                    numbers.append(None)
            present = [n for n in numbers if n is not None]
            if len(present) > 1:
                target = max(present) if better else min(present)
                values = [
                    "**%s**" % value if number == target else value
                    for value, number in zip(values, numbers)
                ]
        lines.append("| %s | %s |" % (name, " | ".join(values)))
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="可重复；PATH 为 run 的 jsonl（自动去重），LABEL 为列名",
    )
    parser.add_argument("--tasks", default=str(Path(__file__).resolve().parents[1] / "query" / "subset_1200_agent_input.json"))
    parser.add_argument("--config", action="append", default=[], metavar="LABEL=JSON", help="每列附带的运行配置")
    parser.add_argument("--no-bold", action="store_true")
    parser.add_argument("--output", help="把表格写到该路径（默认打印）")
    parser.add_argument("--dump-summaries", help="把各列完整summary合并写到该json路径")
    args = parser.parse_args()

    tasks_order = [record["id"] for record in json.load(open(args.tasks, encoding="utf-8"))["records"]]
    configs: Dict[str, Dict[str, Any]] = {}
    for item in args.config:
        label, _, payload = item.partition("=")
        configs[label] = json.loads(payload)

    labels: List[str] = []
    summaries: List[Dict[str, Any]] = []
    for item in args.run:
        label, _, raw_path = item.partition("=")
        if not raw_path:
            raise SystemExit("--run 需要 LABEL=PATH 格式，收到：%s" % item)
        labels.append(label)
        summaries.append(load_summary(Path(raw_path), tasks_order, dict(configs.get(label) or {})))

    table = markdown(labels, summaries, bold_best=not args.no_bold)
    if args.dump_summaries:
        dump_path = Path(args.dump_summaries)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        dump_path.write_text(
            json.dumps({label: summary for label, summary in zip(labels, summaries)}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(table, encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
