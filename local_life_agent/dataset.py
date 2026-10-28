from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List


@dataclass(frozen=True)
class AgentTask:
    id: str
    query: str
    time: str
    loc: Dict[str, float]

    def to_agent_input(self) -> Dict[str, Any]:
        """The only benchmark fields exposed to the agent."""

        return {
            "query": self.query,
            "time": self.time,
            "loc": {"lng": self.loc["lng"], "lat": self.loc["lat"]},
        }


def load_tasks(path: Path) -> List[AgentTask]:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        records = payload.get("records")
    else:
        records = payload
    if not isinstance(records, list):
        raise ValueError("输入数据必须是数组，或包含 records 数组")
    tasks = []
    seen_ids = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError("records[%d] 不是 object" % index)
        context = record.get("context") or {}
        loc = context.get("loc") or {}
        try:
            task = AgentTask(
                id=str(record.get("id") or "row_%04d" % (index + 1)),
                query=str(record["query"]),
                time=str(context["time"]),
                loc={"lng": float(loc["lng"]), "lat": float(loc["lat"])},
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("records[%d] 缺少 query/context.time/context.loc" % index) from exc
        if task.id in seen_ids:
            raise ValueError("数据集中出现重复id: %s" % task.id)
        seen_ids.add(task.id)
        tasks.append(task)
    return tasks
