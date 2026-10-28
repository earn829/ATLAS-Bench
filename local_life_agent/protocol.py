from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


class ProtocolError(ValueError):
    pass


def _repair_inner_quotes(value: str) -> str:
    output: List[str] = []
    in_string = False
    escaped = False
    for index, character in enumerate(value):
        if not in_string:
            output.append(character)
            if character == '"':
                in_string = True
            continue
        if escaped:
            output.append(character)
            escaped = False
            continue
        if character == "\\":
            output.append(character)
            escaped = True
            continue
        if character != '"':
            output.append(character)
            continue
        following = index + 1
        while following < len(value) and value[following].isspace():
            following += 1
        if following == len(value) or value[following] in ",:}]":
            output.append(character)
            in_string = False
        else:
            output.append('\\"')
    return "".join(output)


def _load_json_object(value: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as original_error:
        repaired = _repair_inner_quotes(value)
        if repaired == value:
            raise original_error
        parsed = json.loads(repaired)
    if not isinstance(parsed, dict):
        raise ProtocolError("模型输出顶层必须是 JSON object")
    return parsed


def _extract_json_object(text: str) -> Dict[str, Any]:
    """Parse a single JSON object, tolerating a surrounding Markdown fence."""

    value = text.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        value = "\n".join(lines).strip()
    try:
        return _load_json_object(value)
    except json.JSONDecodeError:
        start = value.find("{")
        end = value.rfind("}")
        if start < 0 or end <= start:
            raise ProtocolError("模型输出中没有合法 JSON 对象")
        try:
            return _load_json_object(value[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ProtocolError("模型输出不是合法 JSON: %s" % exc) from exc


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    tool: str
    arguments: Dict[str, Any]
    rationale: str = ""


@dataclass(frozen=True)
class ToolBatchAction:
    calls: Tuple[ToolCall, ...]


@dataclass(frozen=True)
class FinalAction:
    answer: Dict[str, Any]


def parse_agent_message(text: str, *, max_tools_per_round: int = 2) -> Any:
    payload = _extract_json_object(text)
    message_type = payload.get("type")
    if message_type == "tool_calls":
        calls = payload.get("calls")
        if not isinstance(calls, list):
            raise ProtocolError("tool_calls.calls 必须是数组")
        if not 1 <= len(calls) <= max_tools_per_round:
            raise ProtocolError(
                "tool_calls.calls 每轮必须包含1到%d个调用" % max_tools_per_round
            )
        parsed_calls = []
        seen_call_ids = set()
        for index, call in enumerate(calls):
            if not isinstance(call, dict):
                raise ProtocolError("tool_calls.calls[%d] 必须是 object" % index)
            call_id = call.get("call_id")
            tool = call.get("tool")
            arguments = call.get("arguments")
            if not isinstance(call_id, str) or not call_id.strip():
                raise ProtocolError(
                    "tool_calls.calls[%d].call_id 必须是非空字符串" % index
                )
            call_id = call_id.strip()
            if call_id in seen_call_ids:
                raise ProtocolError("同一轮 call_id 不能重复: %s" % call_id)
            seen_call_ids.add(call_id)
            if not isinstance(tool, str) or not tool.strip():
                raise ProtocolError(
                    "tool_calls.calls[%d].tool 必须是非空字符串" % index
                )
            if not isinstance(arguments, dict):
                raise ProtocolError(
                    "tool_calls.calls[%d].arguments 必须是 JSON object" % index
                )
            rationale = call.get("rationale", "")
            if rationale is None:
                rationale = ""
            if not isinstance(rationale, str):
                raise ProtocolError(
                    "tool_calls.calls[%d].rationale 必须是字符串" % index
                )
            parsed_calls.append(
                ToolCall(
                    call_id=call_id,
                    tool=tool.strip(),
                    arguments=arguments,
                    rationale=rationale,
                )
            )
        return ToolBatchAction(calls=tuple(parsed_calls))
    if message_type == "final":
        answer = payload.get("answer")
        if not isinstance(answer, dict):
            raise ProtocolError("final.answer 必须是 JSON object")
        validate_final_answer(answer)
        return FinalAction(answer=answer)
    raise ProtocolError("type 必须是 tool_calls 或 final")


def validate_final_answer(answer: Dict[str, Any]) -> None:
    recommendations = answer.get("recommendations")
    if not isinstance(recommendations, list):
        raise ProtocolError("answer.recommendations 必须是数组")
    no_exact_match = answer.get("no_exact_match", False)
    if not isinstance(no_exact_match, bool):
        raise ProtocolError("answer.no_exact_match 必须是布尔值")
    seen = set()
    for index, item in enumerate(recommendations, 1):
        if not isinstance(item, dict):
            raise ProtocolError("recommendations[%d] 必须是 object" % (index - 1))
        for field in ("poiid", "name", "reason"):
            if not isinstance(item.get(field), str) or not item[field].strip():
                raise ProtocolError(
                    "recommendations[%d].%s 必须是非空字符串" % (index - 1, field)
                )
        if item["poiid"] in seen:
            raise ProtocolError("推荐结果中出现重复 poiid: %s" % item["poiid"])
        seen.add(item["poiid"])
        if "rank" in item and item["rank"] != index:
            raise ProtocolError("recommendations 的 rank 必须从1连续递增")
        evidence = item.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            raise ProtocolError(
                "recommendations[%d].evidence 必须是非空数组" % (index - 1)
            )
        if any(not isinstance(ref, str) or not ref.strip() for ref in evidence):
            raise ProtocolError(
                "recommendations[%d].evidence 只能包含非空字符串" % (index - 1)
            )
        gaps = item.get("constraint_gaps", [])
        if not isinstance(gaps, list) or any(
            not isinstance(gap, str) or not gap.strip() for gap in gaps
        ):
            raise ProtocolError(
                "recommendations[%d].constraint_gaps 必须是字符串数组" % (index - 1)
            )
    summary = answer.get("summary", "")
    if summary is not None and not isinstance(summary, str):
        raise ProtocolError("answer.summary 必须是字符串")


def json_dumps(value: Any, *, indent: Optional[int] = None) -> str:
    return json.dumps(value, ensure_ascii=False, indent=indent, default=str)
