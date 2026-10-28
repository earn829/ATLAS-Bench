from __future__ import annotations

from pathlib import Path

from .config import AgentConfig, PathConfig
from .protocol import json_dumps
from .tools.registry import ToolRegistry


def build_system_prompt(
    registry: ToolRegistry,
    *,
    paths: PathConfig = PathConfig(),
    config: AgentConfig = AgentConfig(),
) -> str:
    """Load the policy prompt and append the live registry's tool contracts."""

    prompt_path = Path(paths.prompt_file)
    if not prompt_path.exists():
        raise FileNotFoundError("系统提示词不存在: %s" % prompt_path)
    template = prompt_path.read_text(encoding="utf-8").strip()
    runtime_contract = {
        "runtime_defaults": {
            "max_tool_rounds": config.max_tool_rounds,
            "max_tools_per_round": config.max_tools_per_round,
            "max_tool_calls": config.max_tool_calls,
            "default_nearby_distance_m": config.default_nearby_distance_m,
            "max_nearby_distance_m": config.max_nearby_distance_m,
            "default_top_k": config.default_top_k,
            "max_top_k": config.max_top_k,
        },
        "retrieval_baseline": {
            "mode": (
                "qwen3_vector_reranker"
                if config.enable_vector_recall
                else "lexical_fts5_bm25"
            ),
            "candidate_source": "vector_only" if config.enable_vector_recall else "lexical",
            "vector_candidate_k": (
                config.vector_candidate_k if config.enable_vector_recall else None
            ),
            "fusion": None,
            "reranker": bool(config.enable_vector_recall),
        },
        "tools": registry.specifications(),
    }
    return (
        template
        + "\n\n## 运行时工具契约（以此处实时注册内容为准）\n\n```json\n"
        + json_dumps(runtime_contract, indent=2)
        + "\n```"
    )
