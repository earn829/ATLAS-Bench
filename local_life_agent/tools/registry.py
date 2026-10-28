from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List


class ToolInputError(ValueError):
    pass


class Tool(ABC):
    name: str
    description: str
    input_schema: Dict[str, Any]

    def specification(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    @abstractmethod
    def run(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        pass


class ToolRegistry:
    """Generic registry: future tools only need to implement and register ``Tool``."""

    def __init__(self, tools: Iterable[Tool] = ()):
        self._tools: Dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if not tool.name or tool.name in self._tools:
            raise ValueError("重复或空工具名: %s" % tool.name)
        self._tools[tool.name] = tool

    def names(self) -> List[str]:
        return list(self._tools)

    def specifications(self) -> List[Dict[str, Any]]:
        return [tool.specification() for tool in self._tools.values()]

    def execute(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(name, str):
            return {
                "ok": False,
                "tool": str(name),
                "error": "工具名必须是字符串",
                "error_type": "input",
            }
        tool = self._tools.get(name)
        if tool is None:
            return {
                "ok": False,
                "tool": name,
                "error": "未知工具；可用工具: " + ", ".join(self.names()),
            }
        try:
            if not isinstance(arguments, dict):
                raise ToolInputError("arguments 必须是 JSON object")
            schema = tool.input_schema
            required = set(schema.get("required") or [])
            missing = sorted(required - set(arguments))
            if missing:
                raise ToolInputError("缺少必填参数: " + ", ".join(missing))
            if schema.get("additionalProperties") is False:
                allowed = set((schema.get("properties") or {}).keys())
                unknown = sorted(set(arguments) - allowed)
                if unknown:
                    raise ToolInputError("出现未定义参数: " + ", ".join(unknown))
            data = tool.run(arguments)
            return {"ok": True, "tool": name, "data": data}
        except ToolInputError as exc:
            return {"ok": False, "tool": name, "error": str(exc), "error_type": "input"}
        except Exception as exc:
            return {
                "ok": False,
                "tool": name,
                "error": "%s: %s" % (type(exc).__name__, exc),
                "error_type": "runtime",
            }


def required_text(
    arguments: Dict[str, Any], key: str, *, max_length: int = 500
) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ToolInputError("%s 必须是非空字符串" % key)
    value = value.strip()
    if len(value) > max_length:
        raise ToolInputError("%s 最长允许%d个字符" % (key, max_length))
    return value


def bounded_int(
    arguments: Dict[str, Any], key: str, default: int, minimum: int, maximum: int
) -> int:
    value = arguments.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ToolInputError("%s 必须是整数" % key)
    if value < minimum or value > maximum:
        raise ToolInputError("%s 必须在 %d 到 %d 之间" % (key, minimum, maximum))
    return value


def optional_text(
    arguments: Dict[str, Any], key: str, *, max_length: int = 100
) -> str:
    value = arguments.get(key)
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ToolInputError("%s 必须是字符串" % key)
    value = value.strip()
    if len(value) > max_length:
        raise ToolInputError("%s 最长允许%d个字符" % (key, max_length))
    return value


def optional_number(arguments: Dict[str, Any], key: str) -> Any:
    value = arguments.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ToolInputError("%s 必须是数值" % key)
    value = float(value)
    if not math.isfinite(value):
        raise ToolInputError("%s 必须是有限数值" % key)
    return value


def optional_bool(arguments: Dict[str, Any], key: str) -> Any:
    value = arguments.get(key)
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ToolInputError("%s 必须是布尔值" % key)
    return value


def poi_filters(arguments: Dict[str, Any]) -> Dict[str, Any]:
    filters: Dict[str, Any] = {}
    for key in ("category", "district", "business_area"):
        value = optional_text(arguments, key)
        if value:
            filters[key] = value
    for key in ("min_rating", "min_star", "min_average_cost", "max_average_cost"):
        value = optional_number(arguments, key)
        if value is not None:
            if value < 0:
                raise ToolInputError("%s 不能为负数" % key)
            filters[key] = value
    if filters.get("min_rating") is not None and filters["min_rating"] > 5:
        raise ToolInputError("min_rating 不能大于5")
    if filters.get("min_star") is not None and filters["min_star"] > 5:
        raise ToolInputError("min_star 不能大于5")
    min_parking = arguments.get("min_parking_spaces")
    if min_parking is not None:
        if isinstance(min_parking, bool) or not isinstance(min_parking, int):
            raise ToolInputError("min_parking_spaces 必须是整数")
        if min_parking < 0:
            raise ToolInputError("min_parking_spaces 不能为负数")
        filters["min_parking_spaces"] = min_parking
    has_notes = optional_bool(arguments, "has_notes")
    if has_notes is not None:
        filters["has_notes"] = has_notes
    if (
        filters.get("min_average_cost") is not None
        and filters.get("max_average_cost") is not None
        and filters["min_average_cost"] > filters["max_average_cost"]
    ):
        raise ToolInputError("min_average_cost 不能大于 max_average_cost")
    return filters
