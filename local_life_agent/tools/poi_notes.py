from __future__ import annotations

from typing import Any, Dict

from ..retrieval import NoteRepository
from .registry import Tool, bounded_int, required_text


class GetPoiNotesTool(Tool):
    name = "get_poi_notes"
    description = (
        "按精确poiid查询该POI的全部本地笔记，不做近似召回。通常仅在POI召回结果"
        "comment=1且需要核验体验性条件或撰写证据化理由时调用。comment=0只表示本地无笔记。"
    )
    input_schema = {
        "type": "object",
        "required": ["poiid"],
        "properties": {
            "poiid": {"type": "string", "description": "POI唯一ID"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 10, "default": 10},
        },
        "additionalProperties": False,
    }

    def __init__(self, repository: NoteRepository):
        self.repository = repository

    def run(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        poiid = required_text(arguments, "poiid", max_length=256)
        limit = bounded_int(arguments, "limit", 10, 1, 10)
        total_available = self.repository.count_by_poiid(poiid)
        notes = self.repository.by_poiid(poiid, limit=limit)
        for item in notes:
            item["evidence_id"] = "note:" + item["nid"]
        return {
            "poiid": poiid,
            "count": len(notes),
            "total_available": total_available,
            "truncated": total_available > len(notes),
            "notes": notes,
        }
