from __future__ import annotations

from typing import Any, Iterable, Mapping


def _clean_text(parts: Iterable[Any], max_chars: int = 1200) -> str:
    value = "；".join(str(part).strip() for part in parts if str(part or "").strip())
    return value[:max_chars]


def poi_retrieval_text(item: Mapping[str, Any]) -> str:
    return _clean_text(
        (
            "名称：" + str(item.get("name") or ""),
            "类别：" + str(item.get("category") or ""),
            "类别路径：" + str(item.get("category_path") or ""),
            "地址：" + str(item.get("address") or ""),
            "行政区：" + str(item.get("district") or ""),
            "商圈：" + str(item.get("business_area") or ""),
            "设施标签：" + str(item.get("tags") or ""),
        )
    )


def note_retrieval_text(item: Mapping[str, Any]) -> str:
    return _clean_text(
        (
            "商户：" + str(item.get("poi_name") or ""),
            "类别：" + str(item.get("category_path") or ""),
            "体验标签：" + str(item.get("open_tag") or ""),
            "笔记：" + str(item.get("content") or ""),
        )
    )
