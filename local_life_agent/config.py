from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


V5_ROOT = Path(__file__).resolve().parents[1]


def _optional_bool_env(name: str) -> Optional[bool]:
    value = os.environ.get(name, "").strip().lower()
    if not value:
        return None
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError("%s 必须是 true/false" % name)


@dataclass(frozen=True)
class PathConfig:
    """Paths used by the runtime. All generated data stays below ``v5``."""

    poi_db: Path = V5_ROOT / "data" / "indexes" / "poi_rag.sqlite3"
    note_db: Path = V5_ROOT / "data" / "indexes" / "note_rag.sqlite3"
    prompt_file: Path = V5_ROOT / "prompts" / "react_system_prompt.md"
    output_dir: Path = V5_ROOT / "outputs"


@dataclass(frozen=True)
class AgentConfig:
    max_tool_rounds: int = 5
    max_tools_per_round: int = 2
    max_tool_calls: int = 10
    max_parse_retries: int = 2
    default_nearby_distance_m: int = 3000
    max_nearby_distance_m: int = 50000
    default_top_k: int = 10
    max_top_k: int = 20
    max_observation_chars: int = 32000
    enable_vector_recall: bool = True
    vector_required: bool = True
    vector_candidate_k: int = 100
    reranker_batch_size: int = 4
    vector_device: str = "auto"

    def __post_init__(self) -> None:
        if not 1 <= self.max_tool_rounds <= 5:
            raise ValueError("max_tool_rounds 必须在1到5之间")
        if not 1 <= self.max_tools_per_round <= 2:
            raise ValueError("max_tools_per_round 必须在1到2之间")
        if not 1 <= self.max_tool_calls <= 10:
            raise ValueError("max_tool_calls 必须在1到10之间")
        if self.max_parse_retries < 0:
            raise ValueError("max_parse_retries 不能为负")
        if not 1 <= self.default_top_k <= self.max_top_k <= 100:
            raise ValueError("top_k配置必须满足 1 <= default <= max <= 100")
        if not 1 <= self.default_nearby_distance_m <= self.max_nearby_distance_m:
            raise ValueError("附近距离配置必须满足 1 <= default <= max")
        if self.max_observation_chars < 1000:
            raise ValueError("max_observation_chars 不能小于1000")
        if not self.max_top_k <= self.vector_candidate_k <= 1000:
            raise ValueError("vector_candidate_k必须不小于max_top_k且不超过1000")
        if not 1 <= self.reranker_batch_size <= 64:
            raise ValueError("reranker_batch_size必须在1到64之间")
        if self.vector_device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("vector_device 只能是 auto、cpu、cuda 或 mps")


@dataclass(frozen=True)
class LLMConfig:
    """Configuration for an OpenAI-compatible HTTP endpoint."""

    base_url: str
    api_key: str
    model: str
    endpoint_url: str = ""
    api_style: str = "chat_completions"
    timeout_seconds: int = 120
    max_retries: int = 3
    temperature: float = 0.1
    max_output_tokens: int = 4096
    enable_thinking: Optional[bool] = None
    thinking_field: str = "extend_fields"
    reasoning_effort: Optional[str] = None
    repetition_penalty: Optional[float] = None
    omit_temperature: bool = False
    cache_system_prompt: bool = False

    def __post_init__(self) -> None:
        if self.api_style not in {"chat_completions", "responses", "anthropic"}:
            raise ValueError(
                "LOCAL_LIFE_API_STYLE 只能是 chat_completions、responses 或 anthropic"
            )
        if self.timeout_seconds <= 0 or self.max_retries < 1:
            raise ValueError("模型timeout必须为正，max_retries至少为1")
        if self.max_output_tokens < 1:
            raise ValueError("max_output_tokens必须为正")
        if self.thinking_field not in {"extend_fields", "chat_template_kwargs"}:
            raise ValueError(
                "LOCAL_LIFE_THINKING_FIELD 只能是 extend_fields 或 chat_template_kwargs"
            )
        if self.repetition_penalty is not None and not (
            0.0 < self.repetition_penalty <= 2.0
        ):
            raise ValueError("LOCAL_LIFE_REPETITION_PENALTY 必须在 (0, 2] 范围内")

    @classmethod
    def from_env(cls) -> "LLMConfig":
        endpoint_url = (
            os.environ.get("LOCAL_LIFE_API_URL", "").strip()
            or os.environ.get("API_URL", "").strip()
        )
        base_url = (
            os.environ.get("LOCAL_LIFE_API_BASE", "").strip()
            or os.environ.get("OPENAI_BASE_URL", "").strip()
            or endpoint_url
        )
        api_key = (
            os.environ.get("LOCAL_LIFE_API_KEY", "").strip()
            or os.environ.get("API_KEY", "").strip()
            or os.environ.get("OPENAI_API_KEY", "").strip()
        )
        model = (
            os.environ.get("LOCAL_LIFE_MODEL", "").strip()
            or os.environ.get("MODEL_NAME", "").strip()
            or os.environ.get("OPENAI_MODEL_NAME", "").strip()
        )
        if not base_url or not api_key or not model:
            missing = [
                key
                for key, value in (
                    ("LOCAL_LIFE_API_BASE", base_url),
                    ("LOCAL_LIFE_API_KEY", api_key),
                    ("LOCAL_LIFE_MODEL", model),
                )
                if not value
            ]
            raise ValueError("缺少模型环境变量: " + ", ".join(missing))
        api_style = os.environ.get("LOCAL_LIFE_API_STYLE", "").strip()
        if not api_style:
            is_anthropic = (
                "claude" in model.lower()
                or "/protocol/anthropic/" in base_url.lower()
            )
            api_style = "anthropic" if is_anthropic else "chat_completions"
        omit_temperature = _optional_bool_env("LOCAL_LIFE_OMIT_TEMPERATURE")
        if omit_temperature is None:
            omit_temperature = _optional_bool_env("LLM_OMIT_TEMPERATURE")
        return cls(
            base_url=base_url,
            api_key=api_key,
            model=model,
            endpoint_url=endpoint_url,
            api_style=api_style,
            timeout_seconds=int(os.environ.get("LOCAL_LIFE_API_TIMEOUT", "120")),
            max_retries=int(os.environ.get("LOCAL_LIFE_API_RETRIES", "3")),
            temperature=float(os.environ.get("LOCAL_LIFE_TEMPERATURE", "0.1")),
            max_output_tokens=int(
                os.environ.get("LOCAL_LIFE_MAX_OUTPUT_TOKENS", "4096")
            ),
            enable_thinking=_optional_bool_env("LOCAL_LIFE_ENABLE_THINKING"),
            thinking_field=os.environ.get(
                "LOCAL_LIFE_THINKING_FIELD", "extend_fields"
            ).strip()
            or "extend_fields",
            reasoning_effort=(
                os.environ.get("LOCAL_LIFE_REASONING_EFFORT", "").strip() or None
            ),
            repetition_penalty=(
                float(os.environ["LOCAL_LIFE_REPETITION_PENALTY"])
                if os.environ.get("LOCAL_LIFE_REPETITION_PENALTY", "").strip()
                else None
            ),
            omit_temperature=bool(omit_temperature),
            cache_system_prompt=bool(
                _optional_bool_env("LOCAL_LIFE_CACHE_SYSTEM_PROMPT")
            ),
        )
