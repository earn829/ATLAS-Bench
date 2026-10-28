from __future__ import annotations

import json
import time
import http.client
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Sequence

from .config import LLMConfig


class LLMClient(ABC):
    @abstractmethod
    def complete(self, messages: Sequence[Dict[str, str]]) -> Dict[str, Any]:
        """Return ``{"content": str, "usage": dict, "raw": optional}``."""


class OutputBudgetExhausted(RuntimeError):
    """模型在 max_output_tokens 内只产出思考、没有产出 content。

    重发同一请求只会再烧满一次预算，因此不参与重试。
    """


class OpenAICompatibleClient(LLMClient):
    """Dependency-free client for Chat Completions or Responses-style APIs."""

    def __init__(self, config: LLMConfig):
        self.config = config

    def _endpoint(self) -> str:
        if self.config.endpoint_url:
            return self.config.endpoint_url.rstrip("/")
        base = self.config.base_url.rstrip("/")
        suffixes = {
            "responses": "/responses",
            "anthropic": "/messages",
            "chat_completions": "/chat/completions",
        }
        suffix = suffixes[self.config.api_style]
        if base.endswith(suffix):
            return base
        return base + suffix

    def complete(self, messages: Sequence[Dict[str, str]]) -> Dict[str, Any]:
        if self.config.api_style == "responses":
            payload = {
                "model": self.config.model,
                "input": list(messages),
                "max_output_tokens": self.config.max_output_tokens,
                "store": False,
            }
        elif self.config.api_style == "anthropic":
            system_parts = [
                item["content"]
                for item in messages
                if item.get("role") == "system"
            ]
            conversation = [
                {"role": item["role"], "content": item["content"]}
                for item in messages
                if item.get("role") in {"user", "assistant"}
            ]
            payload = {
                "model": self.config.model,
                "messages": conversation,
                "max_tokens": self.config.max_output_tokens,
                # This framework uses a text ReAct protocol. Some Claude gateways
                # otherwise continue by simulating the next user/tool turn.
                "stop_sequences": [
                    "\n\nuser TOOL_OBSERVATION",
                    "\nuser TOOL_OBSERVATION",
                    "\n\nhuman",
                    "\nhuman",
                    "\n\ntest",
                    "\ntest",
                ],
            }
            if system_parts:
                payload["system"] = "\n\n".join(system_parts)
        else:
            chat_messages: List[Dict[str, Any]] = [dict(message) for message in messages]
            if self.config.cache_system_prompt:
                for message in chat_messages:
                    if message.get("role") == "system" and isinstance(
                        message.get("content"), str
                    ):
                        message["content"] = [
                            {
                                "type": "text",
                                "text": message["content"],
                                "cache_control": {"type": "ephemeral"},
                            }
                        ]
                        break
            payload = {
                "model": self.config.model,
                "messages": chat_messages,
                "max_tokens": self.config.max_output_tokens,
            }
            if self.config.enable_thinking is not None:
                template_kwargs = {"enable_thinking": self.config.enable_thinking}
                if self.config.thinking_field == "chat_template_kwargs":
                    payload["chat_template_kwargs"] = template_kwargs
                else:
                    payload["extend_fields"] = {
                        "chat_template_kwargs": template_kwargs
                    }
            if self.config.reasoning_effort is not None:
                payload["reasoning_effort"] = self.config.reasoning_effort
            if self.config.repetition_penalty is not None:
                payload["repetition_penalty"] = self.config.repetition_penalty
        if not self.config.omit_temperature:
            payload["temperature"] = self.config.temperature
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        if self.config.api_style == "anthropic":
            headers = {
                "x-api-key": self.config.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            }
        else:
            headers = {
                "Authorization": "Bearer " + self.config.api_key,
                "Content-Type": "application/json",
            }
        request = urllib.request.Request(
            self._endpoint(),
            data=encoded,
            headers=headers,
            method="POST",
        )
        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                with urllib.request.urlopen(
                    request, timeout=self.config.timeout_seconds
                ) as response:
                    body = json.loads(response.read().decode("utf-8"))
                content = self._extract_content(body)
                return {
                    "content": content,
                    "usage": body.get("usage") or {},
                    "request_id": body.get("id"),
                }
            except (
                urllib.error.URLError,
                urllib.error.HTTPError,
                TimeoutError,
                ValueError,
                # 网关断连时 http.client 直接抛出，urllib 不会包成 URLError；
                # 不重试就会让整条任务连同已完成的工具调用一起报废。
                ConnectionError,
                http.client.HTTPException,
            ) as exc:
                last_error = exc
                if attempt + 1 >= self.config.max_retries:
                    break
                time.sleep(min(2 ** attempt, 8))
        raise RuntimeError("LLM 请求失败: %s" % last_error)

    def _extract_content(self, body: Dict[str, Any]) -> str:
        if self.config.api_style == "anthropic":
            texts = [
                block.get("text", "")
                for block in body.get("content") or []
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            content = "".join(texts)
            if not content:
                raise ValueError("Anthropic 响应缺少 text content block")
            return content
        if self.config.api_style != "responses":
            try:
                choice = body["choices"][0]
                content = choice["message"]["content"]
            except (KeyError, IndexError, TypeError) as exc:
                raise ValueError("Chat Completions 响应缺少 message.content") from exc
            if content is None and choice.get("finish_reason") == "length":
                raise OutputBudgetExhausted(
                    "max_output_tokens=%d 全部被思考占用，未产出 content"
                    % self.config.max_output_tokens
                )
            if not isinstance(content, str):
                raise ValueError("message.content 不是字符串")
            return content
        if isinstance(body.get("output_text"), str):
            return body["output_text"]
        texts = []
        for output in body.get("output") or []:
            for block in output.get("content") or []:
                text = block.get("text")
                if isinstance(text, str):
                    texts.append(text)
        if not texts:
            raise ValueError("Responses 响应缺少文本输出")
        return "\n".join(texts)


class ScriptedLLMClient(LLMClient):
    """Deterministic client used by unit and smoke tests."""

    def __init__(self, responses: Sequence[str]):
        self.responses = list(responses)
        self.calls = []

    def complete(self, messages: Sequence[Dict[str, str]]) -> Dict[str, Any]:
        self.calls.append(list(messages))
        if not self.responses:
            raise RuntimeError("ScriptedLLMClient 没有剩余响应")
        return {"content": self.responses.pop(0), "usage": {}}
