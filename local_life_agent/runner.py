from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Set

from .agent import ReActAgent
from .dataset import AgentTask


def _completed_ids(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    completed: Set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "已有输出 %s 第%d行不是合法JSON，不能安全续跑"
                    % (path, line_number)
                ) from exc
            task_id = record.get("task_id")
            if isinstance(task_id, str) and record.get("status") == "completed":
                completed.add(task_id)
    return completed


def _write_summary(path: Path, summary: Dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def run_batch(
    tasks: Sequence[AgentTask],
    *,
    agent_factory: Callable[[], ReActAgent],
    output_path: Path,
    workers: int = 1,
    resume: bool = True,
) -> Dict[str, Any]:
    """Execute tasks, append records, and aggregate deterministic answer checks."""

    if workers < 1:
        raise ValueError("workers 必须大于等于1")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = _completed_ids(output_path) if resume else set()
    pending = [task for task in tasks if task.id not in existing]
    mode = "a" if resume else "w"
    started = time.time()
    counts = {"completed": 0, "failed": 0}
    judge_counts = {
        "evaluated_answers": 0,
        "passed_answers": 0,
        "no_exact_match_consistent": 0,
        "evidence_reference_count": 0,
        "evidence_known_count": 0,
        "evidence_correct_owner_count": 0,
        "evidence_wrong_owner_count": 0,
        "evidence_unknown_count": 0,
    }
    thread_state = threading.local()

    def account(record: Dict[str, Any]) -> None:
        status = record.get("status", "failed")
        counts[status if status in counts else "failed"] += 1
        judge = record.get("programmatic_judge")
        if not isinstance(judge, dict):
            return
        judge_counts["evaluated_answers"] += 1
        judge_counts["passed_answers"] += int(bool(judge.get("passed")))
        no_exact = judge.get("no_exact_match") or {}
        judge_counts["no_exact_match_consistent"] += int(
            bool(no_exact.get("consistent"))
        )
        ownership = judge.get("evidence_ownership") or {}
        for target, source in (
            ("evidence_reference_count", "reference_count"),
            ("evidence_known_count", "known_count"),
            ("evidence_correct_owner_count", "correct_owner_count"),
            ("evidence_wrong_owner_count", "wrong_owner_count"),
            ("evidence_unknown_count", "unknown_count"),
        ):
            judge_counts[target] += int(ownership.get(source) or 0)

    def execute(task: AgentTask) -> Dict[str, Any]:
        agent = getattr(thread_state, "agent", None)
        if agent is None:
            agent = agent_factory()
            thread_state.agent = agent
        result = agent.run(task.to_agent_input())
        return {"task_id": task.id, **result}

    def safe_execute(task: AgentTask) -> Dict[str, Any]:
        try:
            return execute(task)
        except Exception as exc:
            return {
                "task_id": task.id,
                "status": "failed",
                "answer": None,
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }

    with output_path.open(mode, encoding="utf-8") as handle:
        if workers == 1:
            iterator: Iterable[Dict[str, Any]] = (safe_execute(task) for task in pending)
            for record in iterator:
                account(record)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_task = {
                    executor.submit(safe_execute, task): task for task in pending
                }
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        record = future.result()
                    except Exception as exc:
                        record = {
                            "task_id": task.id,
                            "status": "failed",
                            "answer": None,
                            "error": {
                                "type": type(exc).__name__,
                                "message": str(exc),
                            },
                        }
                    account(record)
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()

    evaluated = judge_counts["evaluated_answers"]
    evidence_total = judge_counts["evidence_reference_count"]
    summary = {
        "output": str(output_path.resolve()),
        "selected_tasks": len(tasks),
        "skipped_existing": len(tasks) - len(pending),
        "executed": len(pending),
        "completed": counts["completed"],
        "failed": counts["failed"],
        "workers": workers,
        "resume": resume,
        "elapsed_seconds": round(time.time() - started, 3),
        "programmatic_judge": {
            "scope": "executed_records",
            **judge_counts,
            "answer_pass_rate": (
                round(judge_counts["passed_answers"] / evaluated, 6)
                if evaluated
                else None
            ),
            "no_exact_match_consistency_rate": (
                round(judge_counts["no_exact_match_consistent"] / evaluated, 6)
                if evaluated
                else None
            ),
            "evidence_existence_accuracy": (
                round(judge_counts["evidence_known_count"] / evidence_total, 6)
                if evidence_total
                else None
            ),
            "evidence_ownership_accuracy": (
                round(
                    judge_counts["evidence_correct_owner_count"] / evidence_total,
                    6,
                )
                if evidence_total
                else None
            ),
        },
    }
    _write_summary(output_path.with_suffix(output_path.suffix + ".summary.json"), summary)
    return summary
