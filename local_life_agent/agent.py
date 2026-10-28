from __future__ import annotations

import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

from .config import AgentConfig, PathConfig
from .llm import LLMClient
from .prompting import build_system_prompt
from .protocol import (
    FinalAction,
    ProtocolError,
    ToolBatchAction,
    ToolCall,
    json_dumps,
    parse_agent_message,
)
from .tools.registry import ToolRegistry


class AgentExecutionError(RuntimeError):
    pass


def _validate_agent_input(value: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("agent输入必须是object")
    expected = {"query", "time", "loc"}
    actual = set(value)
    if actual != expected:
        raise ValueError(
            "agent输入只能且必须包含 query、time、loc；当前多出=%s，缺少=%s"
            % (sorted(actual - expected), sorted(expected - actual))
        )
    query = value.get("query")
    query_time = value.get("time")
    loc = value.get("loc")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query 必须是非空字符串")
    if not isinstance(query_time, str) or not query_time.strip():
        raise ValueError("time 必须是非空字符串")
    if not isinstance(loc, Mapping):
        raise ValueError("loc 必须是 {lng,lat}")
    try:
        lng = float(loc["lng"])
        lat = float(loc["lat"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("loc.lng 和 loc.lat 必须是数值") from exc
    if not math.isfinite(lng) or not math.isfinite(lat):
        raise ValueError("loc 坐标必须是有限数值")
    if not -180 <= lng <= 180 or not -90 <= lat <= 90:
        raise ValueError("loc 坐标越界；顺序必须是 {lng,lat}")
    return {
        "query": query.strip(),
        "time": query_time.strip(),
        "loc": {"lng": lng, "lat": lat},
    }


def _observation_for_model(
    observation: Dict[str, Any], limit: int
) -> Tuple[Dict[str, Any], str]:
    rendered = json_dumps(observation)
    if len(rendered) <= limit:
        return observation, rendered
    visible = json.loads(rendered)
    data = visible.get("data") if isinstance(visible, dict) else None
    lists: List[Tuple[str, List[Any]]] = []
    if isinstance(data, dict):
        for key in ("results", "notes"):
            value = data.get(key)
            if isinstance(value, list):
                lists.append((key, value))
        if lists:
            data["observation_truncated"] = True
            data["original_result_counts"] = {
                key: len(value) for key, value in lists
            }
    while lists:
        rendered = json_dumps(visible)
        if len(rendered) <= limit:
            data["model_visible_counts"] = {
                key: len(value) for key, value in lists
            }
            rendered = json_dumps(visible)
            if len(rendered) <= limit:
                return visible, rendered
        nonempty = [(key, value) for key, value in lists if value]
        if not nonempty:
            break
        _, longest = max(nonempty, key=lambda pair: len(pair[1]))
        longest.pop()
    preview_limit = max(limit - 180, 200)
    fallback = {
        "truncated": True,
        "original_chars": len(json_dumps(observation)),
        "preview": json_dumps(observation)[:preview_limit],
    }
    return fallback, json_dumps(fallback)


def _normalized_name(value: str) -> str:
    return "".join(str(value).lower().split())


class _EvidenceCatalog:
    def __init__(self) -> None:
        self.poi_names: Dict[str, Set[str]] = {}
        self.evidence_owner: Dict[str, str] = {}
        self.successful_observations = 0

    def observe(self, observation: Dict[str, Any]) -> None:
        if not observation.get("ok"):
            return
        self.successful_observations += 1
        data = observation.get("data")
        if not isinstance(data, dict):
            return
        items: List[Any] = []
        for key in ("results", "notes"):
            value = data.get(key)
            if isinstance(value, list):
                items.extend(value)
        for item in items:
            if not isinstance(item, dict):
                continue
            poiid = item.get("poiid")
            if not isinstance(poiid, str) or not poiid.strip():
                continue
            poiid = poiid.strip()
            name = item.get("name") or item.get("poi_name")
            if isinstance(name, str) and name.strip():
                self.poi_names.setdefault(poiid, set()).add(_normalized_name(name))
            evidence_id = item.get("evidence_id")
            if isinstance(evidence_id, str) and evidence_id.strip():
                self.evidence_owner[evidence_id.strip()] = poiid

    def validate_final(self, answer: Dict[str, Any]) -> None:
        errors = []
        if not answer.get("recommendations") and self.successful_observations == 0:
            errors.append("尚未成功调用任何工具，不能直接宣告无匹配结果")
        for item in answer.get("recommendations") or []:
            poiid = item["poiid"].strip()
            if poiid not in self.poi_names:
                errors.append("poiid=%s 从未出现在工具结果中" % poiid)
                continue
            known_names = self.poi_names.get(poiid) or set()
            if known_names and _normalized_name(item["name"]) not in known_names:
                errors.append("poiid=%s 的名称与工具结果不一致" % poiid)
        if errors:
            raise ProtocolError("最终结果未通过POI核验: " + "；".join(errors))

    def judge_final(self, answer: Dict[str, Any]) -> Dict[str, Any]:
        recommendations = answer.get("recommendations") or []
        actual_no_exact = bool(answer.get("no_exact_match", False))
        expected_no_exact = not recommendations
        evidence_checks = []
        known_count = 0
        correct_owner_count = 0
        wrong_owner_count = 0
        unknown_count = 0
        for item in recommendations:
            recommended_poiid = item["poiid"].strip()
            for evidence_id in item.get("evidence") or []:
                owner_poiid = self.evidence_owner.get(evidence_id)
                if owner_poiid is None:
                    status = "unknown"
                    unknown_count += 1
                elif owner_poiid == recommended_poiid:
                    status = "correct_owner"
                    known_count += 1
                    correct_owner_count += 1
                else:
                    status = "wrong_owner"
                    known_count += 1
                    wrong_owner_count += 1
                evidence_checks.append(
                    {
                        "evidence_id": evidence_id,
                        "recommended_poiid": recommended_poiid,
                        "owner_poiid": owner_poiid,
                        "status": status,
                    }
                )
        evidence_count = len(evidence_checks)
        ownership_accuracy = (
            round(correct_owner_count / evidence_count, 6)
            if evidence_count
            else None
        )
        existence_accuracy = (
            round(known_count / evidence_count, 6) if evidence_count else None
        )
        no_exact_consistent = actual_no_exact == expected_no_exact
        evidence_consistent = correct_owner_count == evidence_count
        return {
            "passed": no_exact_consistent and evidence_consistent,
            "no_exact_match": {
                "actual": actual_no_exact,
                "expected": expected_no_exact,
                "consistent": no_exact_consistent,
                "recommendation_count": len(recommendations),
            },
            "evidence_ownership": {
                "reference_count": evidence_count,
                "known_count": known_count,
                "correct_owner_count": correct_owner_count,
                "wrong_owner_count": wrong_owner_count,
                "unknown_count": unknown_count,
                "existence_accuracy": existence_accuracy,
                "ownership_accuracy": ownership_accuracy,
                "consistent": evidence_consistent,
                "checks": evidence_checks,
            },
        }


class ReActAgent:
    """A bounded text-protocol ReAct loop with grounded final recommendations."""

    def __init__(
        self,
        *,
        llm: LLMClient,
        tools: ToolRegistry,
        config: AgentConfig = AgentConfig(),
        system_prompt: Optional[str] = None,
        paths: PathConfig = PathConfig(),
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.config = config
        self.system_prompt = system_prompt or build_system_prompt(
            tools, paths=paths, config=config
        )

    def run(self, agent_input: Mapping[str, Any]) -> Dict[str, Any]:
        safe_input = _validate_agent_input(agent_input)
        started = time.time()
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    "以下是本次唯一可见的 benchmark 输入。请开始检索并回答。\n"
                    + json_dumps(safe_input, indent=2)
                ),
            },
        ]
        trajectory: List[Dict[str, Any]] = []
        usage_events: List[Dict[str, Any]] = []
        evidence = _EvidenceCatalog()
        model_calls = 0
        tool_rounds = 0
        tool_calls = 0

        def next_action(
            *, tools_allowed: bool, max_tools_this_round: int = 0
        ) -> Tuple[Any, str]:
            nonlocal model_calls
            last_error: Optional[Exception] = None
            for retry in range(self.config.max_parse_retries + 1):
                response = self.llm.complete(messages)
                model_calls += 1
                raw = response.get("content")
                if not isinstance(raw, str):
                    raw = ""
                usage = response.get("usage")
                if isinstance(usage, dict):
                    usage_events.append(usage)
                try:
                    action = parse_agent_message(
                        raw,
                        max_tools_per_round=(
                            max_tools_this_round
                            if tools_allowed
                            else self.config.max_tools_per_round
                        ),
                    )
                    if not tools_allowed and isinstance(action, ToolBatchAction):
                        raise ProtocolError("工具调用预算已用尽，此轮只能输出 final")
                    if isinstance(action, FinalAction):
                        evidence.validate_final(action.answer)
                    return action, raw
                except ProtocolError as exc:
                    last_error = exc
                    trajectory.append(
                        {
                            "event": "protocol_error",
                            "model_call": model_calls,
                            "retry": retry,
                            "error": str(exc),
                            "raw_output": raw,
                        }
                    )
                    allowed = (
                        "tool_calls（calls包含1到%d项）或 final"
                        % max_tools_this_round
                        if tools_allowed
                        else "final"
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "PROTOCOL_ERROR: %s\n上一回复未被执行，其中自行续写的工具观察"
                                "或证据一律无效，不得继续引用。请修正。只能输出一个合法 JSON 对象，"
                                "本轮允许的 type 是 %s。不要添加解释或 Markdown。"
                                % (exc, allowed)
                            ),
                        }
                    )
            raise AgentExecutionError(
                "模型连续 %d 次未遵守协议: %s"
                % (self.config.max_parse_retries + 1, last_error)
            )

        def completed(answer: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "status": "completed",
                "input": safe_input,
                "answer": answer,
                "programmatic_judge": evidence.judge_final(answer),
                "trajectory": trajectory,
                "metrics": {
                    "model_calls": model_calls,
                    "tool_rounds": tool_rounds,
                    "tool_calls": tool_calls,
                    "elapsed_seconds": round(time.time() - started, 3),
                    "usage_events": usage_events,
                },
            }

        def execute_one(call: ToolCall) -> Tuple[Dict[str, Any], float]:
            tool_started = time.time()
            observation = self.tools.execute(call.tool, call.arguments)
            return observation, round(time.time() - tool_started, 4)

        try:
            for decision_round in range(1, self.config.max_tool_rounds + 1):
                remaining_tool_calls = self.config.max_tool_calls - tool_calls
                if remaining_tool_calls <= 0:
                    break
                max_tools_this_round = min(
                    self.config.max_tools_per_round, remaining_tool_calls
                )
                action, raw = next_action(
                    tools_allowed=True,
                    max_tools_this_round=max_tools_this_round,
                )
                if isinstance(action, FinalAction):
                    trajectory.append(
                        {
                            "event": "final",
                            "decision_round": decision_round,
                            "after_tool_round": tool_rounds,
                            "raw_output": raw,
                        }
                    )
                    return completed(action.answer)
                messages.append({"role": "assistant", "content": raw})
                batch_started = time.time()
                if len(action.calls) == 1:
                    execution_results = [execute_one(action.calls[0])]
                else:
                    with ThreadPoolExecutor(max_workers=len(action.calls)) as executor:
                        execution_results = list(executor.map(execute_one, action.calls))

                first_tool_step = tool_calls + 1
                tool_calls += len(action.calls)
                tool_rounds += 1
                per_observation_limit = max(
                    500, self.config.max_observation_chars // len(action.calls)
                )
                visible_results = []
                trajectory_calls = []
                for index, (call, execution_result) in enumerate(
                    zip(action.calls, execution_results)
                ):
                    observation, elapsed = execution_result
                    visible_observation, _ = _observation_for_model(
                        observation, per_observation_limit
                    )
                    evidence.observe(visible_observation)
                    visible_results.append(
                        {
                            "call_id": call.call_id,
                            "tool": call.tool,
                            "observation": visible_observation,
                        }
                    )
                    trajectory_calls.append(
                        {
                            "step": first_tool_step + index,
                            "call_id": call.call_id,
                            "tool": call.tool,
                            "arguments": call.arguments,
                            "rationale": call.rationale,
                            "observation": observation,
                            "observation_truncated_for_model": (
                                visible_observation is not observation
                            ),
                            "elapsed_seconds": elapsed,
                        }
                    )
                trajectory.append(
                    {
                        "event": "tool_calls",
                        "round": tool_rounds,
                        "parallel": len(action.calls) > 1,
                        "calls": trajectory_calls,
                        "elapsed_seconds": round(time.time() - batch_started, 4),
                    }
                )
                messages.append(
                    {
                        "role": "user",
                        "content": "TOOL_OBSERVATIONS\n"
                        + json_dumps(
                            {
                                "tool_round": tool_rounds,
                                "observations": visible_results,
                            }
                        ),
                    }
                )

            messages.append(
                {
                    "role": "user",
                    "content": (
                        "工具调用预算已用尽。请仅依据已经看到的工具证据立即输出 final；"
                        "如果没有合格候选，recommendations=[] 且 no_exact_match=true。"
                    ),
                }
            )
            action, raw = next_action(tools_allowed=False)
            if not isinstance(action, FinalAction):
                raise AgentExecutionError("工具预算耗尽后模型仍未返回 final")
            trajectory.append(
                {
                    "event": "final",
                    "after_tool_round": tool_rounds,
                    "forced": True,
                    "raw_output": raw,
                }
            )
            return completed(action.answer)
        except Exception as exc:
            trajectory.append(
                {"event": "failed", "error_type": type(exc).__name__, "error": str(exc)}
            )
            return {
                "status": "failed",
                "input": safe_input,
                "answer": None,
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "trajectory": trajectory,
                "metrics": {
                    "model_calls": model_calls,
                    "tool_rounds": tool_rounds,
                    "tool_calls": tool_calls,
                    "elapsed_seconds": round(time.time() - started, 3),
                    "usage_events": usage_events,
                },
            }
