from __future__ import annotations

from typing import Any, Dict

from ..retrieval import PoiRepository
from .registry import Tool, bounded_int, poi_filters, required_text


class SearchPoisTool(Tool):
    name = "search_pois"
    description = (
        "从POI结构化RAG库以Qwen3纯向量召回地点或商家候选，并经Qwen3 reranker重排。适合查名称、类别、地址、商圈、评分、"
        "人均、营业时间、设施标签，也可用于先解析命名地标的坐标。结果中的comment是"
        "0/1标志：1表示该POI至少有一条笔记，可继续调用get_poi_notes。"
    )
    input_schema = {
        "type": "object",
        "required": ["query"],
        "properties": {
            "query": {"type": "string", "maxLength": 500, "description": "POI召回文本或精确poiid"},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 20, "default": 10},
            "category": {"type": "string", "description": "类别/类别路径包含过滤"},
            "district": {"type": "string", "description": "行政区包含过滤"},
            "business_area": {"type": "string", "description": "商圈包含过滤"},
            "min_rating": {"type": "number", "minimum": 0, "maximum": 5, "description": "最低评分"},
            "min_star": {"type": "number", "minimum": 0, "maximum": 5, "description": "最低星级"},
            "min_average_cost": {"type": "number", "minimum": 0, "description": "最低人均"},
            "max_average_cost": {"type": "number", "minimum": 0, "description": "最高人均"},
            "min_parking_spaces": {"type": "integer", "minimum": 0, "description": "最少停车位数"},
            "has_notes": {"type": "boolean", "description": "是否必须有非空笔记"},
        },
        "additionalProperties": False,
    }

    def __init__(
        self, repository: PoiRepository, max_top_k: int = 20, default_top_k: int = 10
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
        filters = poi_filters(arguments)
        results = self.repository.search(query, top_k=top_k, filters=filters)
        for item in results:
            item["evidence_id"] = "poi:" + item["poiid"]
        return {
            "query": query,
            "retrieval_mode": getattr(
                self.repository, "retrieval_mode", "lexical_fts5_bm25"
            ),
            "applied_filters": filters,
            "count": len(results),
            "results": results,
        }
