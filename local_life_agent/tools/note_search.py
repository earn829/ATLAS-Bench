from __future__ import annotations

from typing import Any, Dict

from ..retrieval import NoteRepository
from .registry import Tool, bounded_int, optional_text, required_text


class SearchNotesTool(Tool):
    name = "search_notes"
    description = (
        "从全局笔记RAG库以Qwen3纯向量召回用户体验证据候选，并经Qwen3 reranker重排。适合安静、服务、正宗、适合带娃、环境、"
        "口碑等不能由POI结构字段直接证明的条件。返回nid、poiid、正文和标签。"
    )
    input_schema = {
        "type": "object",
        "required": ["query"],
        "properties": {
            "query": {"type": "string", "maxLength": 500, "description": "笔记内容召回文本"},
            "category": {"type": "string", "maxLength": 100, "description": "笔记关联POI类别路径包含过滤"},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 20, "default": 10},
        },
        "additionalProperties": False,
    }

    def __init__(
        self, repository: NoteRepository, max_top_k: int = 20, default_top_k: int = 10
    ):
        if not 1 <= default_top_k <= max_top_k:
            raise ValueError("default_top_k必须在1到max_top_k之间")
        self.repository = repository
        self.max_top_k = max_top_k
        self.default_top_k = default_top_k
        self.input_schema = {
            **type(self).input_schema,
            "properties": {
                **type(self).input_schema["properties"],
                "top_k": {
                    **type(self).input_schema["properties"]["top_k"],
                    "maximum": max_top_k,
                    "default": default_top_k,
                },
            },
        }

    def run(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        query = required_text(arguments, "query")
        top_k = bounded_int(
            arguments, "top_k", self.default_top_k, 1, self.max_top_k
        )
        category = optional_text(arguments, "category")
        results = self.repository.search(query, top_k=top_k, category=category)
        for item in results:
            item["evidence_id"] = "note:" + item["nid"]
        return {
            "query": query,
            "retrieval_mode": getattr(
                self.repository, "retrieval_mode", "lexical_fts5_bm25"
            ),
            "applied_filters": {"category": category} if category else {},
            "per_poi_limit": 2,
            "count": len(results),
            "results": results,
        }
