from __future__ import annotations

import math
from typing import Any, Dict

from ..retrieval import PoiRepository
from .registry import Tool, ToolInputError, bounded_int, poi_filters, required_text


class SearchNearbyTool(Tool):
    name = "search_nearby"
    description = (
        "在给定发起点坐标和距离半径内以Qwen3纯向量召回POI候选，并经Qwen3 reranker重排。用空间索引和Haversine距离"
        "精确过滤，保证每条结果distance_m不超过distance_m参数。必须同时传query、"
        "loc{lng,lat}和distance_m，不能只把‘附近’写进文本召回。"
    )
    input_schema = {
        "type": "object",
        "required": ["query", "loc", "distance_m"],
        "properties": {
            "query": {"type": "string", "description": "要搜索的POI文本"},
            "loc": {
                "type": "object",
                "required": ["lng", "lat"],
                "properties": {"lng": {"type": "number"}, "lat": {"type": "number"}},
                "additionalProperties": False,
            },
            "distance_m": {
                "type": "number",
                "minimum": 1,
                "maximum": 50000,
                "description": "搜索半径，单位米",
            },
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
        self,
        repository: PoiRepository,
        max_distance_m: int = 50000,
        max_top_k: int = 20,
        default_top_k: int = 10,
    ):
        if not 1 <= default_top_k <= max_top_k:
            raise ValueError("default_top_k必须在1到max_top_k之间")
        self.repository = repository
        self.max_distance_m = max_distance_m
        self.max_top_k = max_top_k
        self.default_top_k = default_top_k
        properties = {
            **type(self).input_schema["properties"],
            "distance_m": {
                **type(self).input_schema["properties"]["distance_m"],
                "maximum": max_distance_m,
            },
            "top_k": {
                **type(self).input_schema["properties"]["top_k"],
                "maximum": max_top_k,
                "default": default_top_k,
            },
        }
        self.input_schema = {**type(self).input_schema, "properties": properties}

    def run(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        query = required_text(arguments, "query")
        loc = arguments.get("loc")
        if not isinstance(loc, dict):
            raise ToolInputError("loc 必须是包含lng和lat的object")
        if set(loc) != {"lng", "lat"}:
            raise ToolInputError("loc 只能且必须包含lng和lat")
        try:
            raw_lng = loc["lng"]
            raw_lat = loc["lat"]
            raw_distance = arguments["distance_m"]
            if any(isinstance(value, bool) for value in (raw_lng, raw_lat, raw_distance)):
                raise TypeError
            if any(not isinstance(value, (int, float)) for value in (raw_lng, raw_lat, raw_distance)):
                raise TypeError
            lng = float(raw_lng)
            lat = float(raw_lat)
            distance_m = float(raw_distance)
        except (KeyError, TypeError, ValueError) as exc:
            raise ToolInputError("loc.lng、loc.lat、distance_m 必须是数值") from exc
        if not all(math.isfinite(value) for value in (lng, lat, distance_m)):
            raise ToolInputError("loc.lng、loc.lat、distance_m 必须是有限数值")
        if not -180 <= lng <= 180 or not -90 <= lat <= 90:
            raise ToolInputError("经纬度越界；坐标顺序必须是{lng,lat}")
        if distance_m < 1 or distance_m > self.max_distance_m:
            raise ToolInputError("distance_m 必须在1到%d米之间" % self.max_distance_m)
        top_k = bounded_int(
            arguments, "top_k", self.default_top_k, 1, self.max_top_k
        )
        filters = poi_filters(arguments)
        results = self.repository.nearby(
            query=query,
            lng=lng,
            lat=lat,
            distance_m=distance_m,
            top_k=top_k,
            filters=filters,
        )
        for item in results:
            item["evidence_id"] = "poi:" + item["poiid"]
        return {
            "query": query,
            "retrieval_mode": getattr(
                self.repository, "retrieval_mode", "lexical_fts5_bm25"
            ),
            "origin": {"lng": lng, "lat": lat},
            "distance_m": distance_m,
            "applied_filters": filters,
            "count": len(results),
            "results": results,
        }
