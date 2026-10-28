from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from .dense_retrieval import FaissVectorIndex
from .reranking import SharedQwen3Reranker
from .retrieval import NoteRepository, PoiRepository
from .text_representation import note_retrieval_text, poi_retrieval_text


def _rerank(
    query: str,
    items: Sequence[Dict[str, Any]],
    *,
    reranker: SharedQwen3Reranker,
    text_builder,
    key_field: str,
) -> List[Dict[str, Any]]:
    if not items:
        return []
    scores = reranker.score(query, [text_builder(item) for item in items])
    if len(scores) != len(items):
        raise RuntimeError("reranker返回分数数量与候选数量不一致")
    scored = []
    for item, score in zip(items, scores):
        copied = dict(item)
        copied.pop("_vector_row_id", None)
        copied["rerank_score"] = round(float(score), 8)
        scored.append(copied)
    scored.sort(
        key=lambda item: (
            -item["rerank_score"],
            item.get("vector_rank", 10**9),
            str(item[key_field]),
        )
    )
    for rank, item in enumerate(scored, 1):
        item["rerank_rank"] = rank
    return scored


class VectorRerankPoiRepository:
    retrieval_mode = "qwen3_vector_reranker"

    def __init__(
        self,
        store: PoiRepository,
        vector: FaissVectorIndex,
        reranker: SharedQwen3Reranker,
        *,
        candidate_k: int,
    ) -> None:
        self.store = store
        self.vector = vector
        self.reranker = reranker
        self.candidate_k = int(candidate_k)
        self.path = store.path
        self.build_id = store.build_id

    def _vector_results(
        self,
        query: str,
        *,
        eligible_ids: Optional[Sequence[int]] = None,
    ) -> List[Dict[str, Any]]:
        ranked = self.vector.search_ids(
            query,
            self.candidate_k,
            eligible_ids=eligible_ids,
        )
        hydrated = self.store.by_row_ids([identifier for identifier, _ in ranked])
        output = []
        for rank, (identifier, similarity) in enumerate(ranked, 1):
            item = hydrated.get(identifier)
            if item is None:
                continue
            item["_vector_row_id"] = identifier
            item["vector_rank"] = rank
            item["vector_similarity"] = round(float(similarity), 8)
            output.append(item)
        return output

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Mapping[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        filters = filters or {}
        exact = self.store.get_matching(query.strip(), filters=filters)
        if exact is not None:
            exact.update(
                {
                    "retrieval_stage": "exact_poiid_lookup",
                    "vector_rank": None,
                    "vector_similarity": None,
                    "rerank_rank": 1,
                    "rerank_score": None,
                }
            )
            return [exact]
        eligible_ids = self.store.filtered_row_ids(filters) if filters else None
        vector_items = self._vector_results(query, eligible_ids=eligible_ids)
        return _rerank(
            query,
            vector_items,
            reranker=self.reranker,
            text_builder=poi_retrieval_text,
            key_field="poiid",
        )[:top_k]

    def get(self, poiid: str) -> Optional[Dict[str, Any]]:
        return self.store.get(poiid)

    def nearby(
        self,
        *,
        query: str,
        lng: float,
        lat: float,
        distance_m: float,
        top_k: int,
        filters: Optional[Mapping[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        distances = self.store.nearby_row_distances(
            lng=lng,
            lat=lat,
            distance_m=distance_m,
            filters=filters or {},
        )
        vector_items = self._vector_results(
            query,
            eligible_ids=list(distances),
        )
        for item in vector_items:
            row_id = item.pop("_vector_row_id")
            item["distance_m"] = round(float(distances[row_id]), 1)
        results = _rerank(
            query,
            vector_items,
            reranker=self.reranker,
            text_builder=poi_retrieval_text,
            key_field="poiid",
        )[:top_k]
        for item in results:
            item.pop("_vector_row_id", None)
        return results


class VectorRerankNoteRepository:
    retrieval_mode = "qwen3_vector_reranker"

    def __init__(
        self,
        store: NoteRepository,
        vector: FaissVectorIndex,
        reranker: SharedQwen3Reranker,
        *,
        candidate_k: int,
    ) -> None:
        self.store = store
        self.vector = vector
        self.reranker = reranker
        self.candidate_k = int(candidate_k)
        self.path = store.path
        self.build_id = store.build_id

    def search(
        self, query: str, top_k: int = 10, category: str = ""
    ) -> List[Dict[str, Any]]:
        eligible_ids = self.store.filtered_row_ids(category=category) if category else None
        ranked = self.vector.search_ids(
            query,
            self.candidate_k,
            eligible_ids=eligible_ids,
        )
        hydrated = self.store.by_row_ids(
            [identifier for identifier, _ in ranked],
            max_content_chars=1200,
        )
        vector_items = []
        for rank, (identifier, similarity) in enumerate(ranked, 1):
            item = hydrated.get(identifier)
            if item is None:
                continue
            item["vector_rank"] = rank
            item["vector_similarity"] = round(float(similarity), 8)
            vector_items.append(item)
        reranked = _rerank(
            query,
            vector_items,
            reranker=self.reranker,
            text_builder=note_retrieval_text,
            key_field="nid",
        )
        diversified = []
        per_poi_counts: Dict[str, int] = {}
        for item in reranked:
            poiid = item["poiid"]
            if per_poi_counts.get(poiid, 0) >= 2:
                continue
            per_poi_counts[poiid] = per_poi_counts.get(poiid, 0) + 1
            content = str(item.get("content") or "")
            if len(content) > 600:
                item["content"] = content[:600].rstrip() + "…"
                item["content_truncated"] = True
            diversified.append(item)
            if len(diversified) >= top_k:
                break
        for rank, item in enumerate(diversified, 1):
            item["rerank_rank"] = rank
        return diversified

    def count_by_poiid(self, poiid: str) -> int:
        return self.store.count_by_poiid(poiid)

    def by_poiid(self, poiid: str, limit: int = 10) -> List[Dict[str, Any]]:
        return self.store.by_poiid(poiid, limit=limit)
