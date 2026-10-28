from __future__ import annotations

import heapq
import json
import math
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import quote

from .index_builder import SCHEMA_VERSION


_SEGMENT_RE = re.compile(r"[\u3400-\u9fff]+|[a-zA-Z0-9]+")
_SPACE_RE = re.compile(r"\s+")
_POI_FIELDS = (
    "id",
    "poiid",
    "name",
    "latitude",
    "longitude",
    "address",
    "pcdr",
    "district",
    "business_area",
    "category",
    "category_path",
    "rating",
    "average_cost",
    "star",
    "open_time",
    "tags",
    "tickets",
    "mall_hotel_hours_flag",
    "parking_spaces",
    "comment",
    "comment_count",
    "search_text",
)
_NOTE_FIELDS = (
    "id",
    "nid",
    "poiid",
    "poi_name",
    "content",
    "created_at",
    "modified_at",
    "category_path",
    "open_tag",
    "open_tag_term",
    "open_tag_detail",
    "hot_score",
    "like_num",
    "view_num",
    "share_num",
    "reply_num",
    "favorites_num",
)


def normalize_query(text: str) -> str:
    return _SPACE_RE.sub(" ", str(text or "").strip().lower())


def _query_tokens(text: str, max_terms: int = 96) -> List[str]:
    tokens: List[str] = []
    seen = set()
    for segment in _SEGMENT_RE.findall(normalize_query(text)):
        is_cjk = "\u3400" <= segment[0] <= "\u9fff"
        if is_cjk:
            if len(segment) == 1:
                candidates: Sequence[str] = []
            elif len(segment) == 2:
                candidates = [segment]
            else:
                candidates = [segment[index : index + 2] for index in range(len(segment) - 1)]
                if len(segment) <= 16:
                    candidates = list(candidates) + [segment]
        else:
            candidates = [segment]
            if len(segment) > 4:
                candidates = list(candidates) + [
                    segment[index : index + 3] for index in range(len(segment) - 2)
                ]
        for token in candidates:
            if token and token not in seen:
                seen.add(token)
                tokens.append(token)
    if len(tokens) > max_terms:
        half = max_terms // 2
        tokens = tokens[:half] + tokens[-(max_terms - half) :]
    return tokens


def _fts_query(text: str) -> Optional[str]:
    terms = _query_tokens(text)
    if not terms:
        return None
    return " OR ".join('"%s"' % term.replace('"', '""') for term in terms)


def _columns(alias: str, fields: Sequence[str]) -> str:
    return ", ".join("%s.%s AS %s" % (alias, field, field) for field in fields)


@contextmanager
def _connection(
    path: Path, expected_build_id: Optional[str] = None
) -> Iterator[sqlite3.Connection]:
    resolved = Path(path).resolve()
    uri = "file:%s?mode=ro" % quote(resolved.as_posix(), safe="/")
    connection = sqlite3.connect(uri, uri=True, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    try:
        if expected_build_id is not None:
            row = connection.execute(
                "SELECT value FROM metadata WHERE key='build_id'"
            ).fetchone()
            actual_build_id = row[0] if row else None
            if actual_build_id != expected_build_id:
                raise RuntimeError(
                    "索引构建版本不一致: expected=%s actual=%s"
                    % (expected_build_id, actual_build_id)
                )
        yield connection
    finally:
        connection.close()


def _check_database(
    path: Path,
    required_tables: Sequence[str],
    label: str,
    expected_build_id: Optional[str] = None,
) -> str:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("%s不存在: %s" % (label, path))
    with _connection(path) as connection:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        if version != SCHEMA_VERSION:
            raise RuntimeError(
                "%s schema版本=%s，运行时要求=%s；请重建索引"
                % (label, version, SCHEMA_VERSION)
            )
        names = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table','view')"
            )
        }
        missing = set(required_tables) | {"metadata"}
        missing -= names
        if missing:
            raise RuntimeError("%s缺少表: %s" % (label, sorted(missing)))
        row = connection.execute(
            "SELECT value FROM metadata WHERE key='build_id'"
        ).fetchone()
        build_id = row[0] if row else ""
        if not build_id:
            raise RuntimeError("%s缺少build_id" % label)
        if expected_build_id is not None and build_id != expected_build_id:
            raise RuntimeError(
                "%s与manifest构建版本不一致: expected=%s actual=%s"
                % (label, expected_build_id, build_id)
            )
        return build_id


def _like_pattern(value: str) -> str:
    escaped = normalize_query(value).replace("\\", "\\\\")
    escaped = escaped.replace("%", "\\%").replace("_", "\\_")
    return "%" + escaped + "%"


def _filter_sql(filters: Mapping[str, Any], alias: str = "p") -> Tuple[str, List[Any]]:
    clauses: List[str] = []
    params: List[Any] = []
    text_fields = {
        "category": ("category", "category_path"),
        "district": ("district",),
        "business_area": ("business_area",),
    }
    for key, fields in text_fields.items():
        value = filters.get(key)
        if value:
            subclauses = []
            for field in fields:
                subclauses.append("lower(%s.%s) LIKE ? ESCAPE '\\'" % (alias, field))
                params.append(_like_pattern(str(value)))
            clauses.append("(" + " OR ".join(subclauses) + ")")
    for key, operator, column in (
        ("min_rating", ">=", "rating"),
        ("min_star", ">=", "star"),
        ("min_average_cost", ">=", "average_cost"),
        ("max_average_cost", "<=", "average_cost"),
        ("min_parking_spaces", ">=", "parking_spaces"),
    ):
        value = filters.get(key)
        if value is not None:
            clauses.append("%s.%s %s ?" % (alias, column, operator))
            params.append(value)
    if filters.get("has_notes") is not None:
        clauses.append("%s.comment = ?" % alias)
        params.append(int(bool(filters["has_notes"])))
    return (" AND ".join(clauses), params)


def _fallback_text_sql(query: str, alias: str, fields: Sequence[str]) -> Tuple[str, List[Any]]:
    segments = _SEGMENT_RE.findall(normalize_query(query)) or [normalize_query(query)]
    clauses: List[str] = []
    params: List[Any] = []
    for segment in segments[:20]:
        if not segment:
            continue
        field_clauses = []
        for field in fields:
            field_clauses.append("lower(%s.%s) LIKE ? ESCAPE '\\'" % (alias, field))
            params.append(_like_pattern(segment))
        clauses.append("(" + " OR ".join(field_clauses) + ")")
    return (" OR ".join(clauses) or "0", params)


def _validate_top_k(top_k: int) -> int:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 100:
        raise ValueError("top_k 必须是1到100之间的整数")
    return top_k


def _poi_dict(row: sqlite3.Row, *, include_payload: bool = False) -> Dict[str, Any]:
    item = {
        "poiid": row["poiid"],
        "name": row["name"],
        "category": row["category"],
        "category_path": row["category_path"],
        "address": row["address"],
        "district": row["district"],
        "business_area": row["business_area"],
        "loc": {"lng": row["longitude"], "lat": row["latitude"]},
        "rating": row["rating"],
        "average_cost": row["average_cost"],
        "star": row["star"],
        "open_time": row["open_time"],
        "tags": row["tags"],
        "tickets": row["tickets"],
        "mall_hotel_hours_flag": row["mall_hotel_hours_flag"],
        "parking_spaces": row["parking_spaces"],
        "comment": int(row["comment"]),
        "comment_count": int(row["comment_count"]),
    }
    if "retrieval_rank" in row.keys():
        item["retrieval_score"] = round(-float(row["retrieval_rank"]), 8)
    if include_payload and "payload_json" in row.keys():
        item["payload"] = json.loads(row["payload_json"])
    return item


class PoiRepository:
    def __init__(self, path: Path, expected_build_id: Optional[str] = None):
        self.path = Path(path)
        self.build_id = _check_database(
            self.path,
            ("pois", "poi_fts", "poi_geo"),
            "POI RAG库",
            expected_build_id,
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Mapping[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        top_k = _validate_top_k(top_k)
        filters = filters or {}
        filter_sql, filter_params = _filter_sql(filters)
        filter_suffix = (" AND " + filter_sql) if filter_sql else ""
        columns = _columns("p", _POI_FIELDS)
        with _connection(self.path, self.build_id) as connection:
            exact = connection.execute(
                "SELECT %s, 0.0 AS retrieval_rank FROM pois p "
                "WHERE p.poiid = ?%s LIMIT 1" % (columns, filter_suffix),
                [query.strip()] + filter_params,
            ).fetchone()
            if exact is not None:
                return [_poi_dict(exact)]
            match = _fts_query(query)
            if match:
                rows = connection.execute(
                    """
                    SELECT %s, bm25(poi_fts, 8.0, 4.0, 1.0) AS retrieval_rank
                    FROM poi_fts JOIN pois p ON p.id = poi_fts.rowid
                    WHERE poi_fts MATCH ?%s
                    ORDER BY retrieval_rank, p.id
                    LIMIT ?
                    """
                    % (columns, filter_suffix),
                    [match] + filter_params + [top_k],
                ).fetchall()
            else:
                text_sql, text_params = _fallback_text_sql(
                    query, "p", ("name", "category", "category_path", "search_text")
                )
                where = "(" + text_sql + ")"
                if filter_sql:
                    where += " AND " + filter_sql
                rows = connection.execute(
                    "SELECT %s, 0.0 AS retrieval_rank FROM pois p "
                    "WHERE %s ORDER BY p.id LIMIT ?" % (columns, where),
                    text_params + filter_params + [top_k],
                ).fetchall()
        return [_poi_dict(row) for row in rows]

    def get(self, poiid: str) -> Optional[Dict[str, Any]]:
        columns = _columns("p", _POI_FIELDS)
        with _connection(self.path, self.build_id) as connection:
            row = connection.execute(
                "SELECT %s, p.payload_json AS payload_json FROM pois p WHERE p.poiid = ?"
                % columns,
                (poiid,),
            ).fetchone()
        return _poi_dict(row, include_payload=True) if row else None

    def get_matching(
        self, poiid: str, filters: Optional[Mapping[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        filter_sql, filter_params = _filter_sql(filters or {})
        suffix = (" AND " + filter_sql) if filter_sql else ""
        columns = _columns("p", _POI_FIELDS)
        with _connection(self.path, self.build_id) as connection:
            row = connection.execute(
                "SELECT %s FROM pois p WHERE p.poiid = ?%s LIMIT 1"
                % (columns, suffix),
                [poiid] + filter_params,
            ).fetchone()
        return _poi_dict(row) if row else None

    def filtered_row_ids(
        self, filters: Optional[Mapping[str, Any]] = None
    ) -> List[int]:
        filter_sql, filter_params = _filter_sql(filters or {})
        where = (" WHERE " + filter_sql) if filter_sql else ""
        with _connection(self.path, self.build_id) as connection:
            rows = connection.execute(
                "SELECT p.id FROM pois p%s ORDER BY p.id" % where,
                filter_params,
            ).fetchall()
        return [int(row[0]) for row in rows]

    def by_row_ids(
        self,
        row_ids: Sequence[int],
        filters: Optional[Mapping[str, Any]] = None,
    ) -> Dict[int, Dict[str, Any]]:
        identifiers = [int(value) for value in row_ids]
        if not identifiers:
            return {}
        placeholders = ",".join("?" for _ in identifiers)
        filter_sql, filter_params = _filter_sql(filters or {})
        suffix = (" AND " + filter_sql) if filter_sql else ""
        columns = _columns("p", _POI_FIELDS)
        with _connection(self.path, self.build_id) as connection:
            rows = connection.execute(
                "SELECT %s FROM pois p WHERE p.id IN (%s)%s"
                % (columns, placeholders, suffix),
                identifiers + filter_params,
            ).fetchall()
        return {int(row["id"]): _poi_dict(row) for row in rows}

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
        top_k = _validate_top_k(top_k)
        if not all(math.isfinite(value) for value in (lng, lat, distance_m)):
            raise ValueError("经纬度和distance_m必须是有限数值")
        if not -180 <= lng <= 180 or not -90 <= lat <= 90 or distance_m <= 0:
            raise ValueError("经纬度越界或distance_m非正数")
        angular = distance_m / 6371008.8
        lat_delta = math.degrees(angular)
        lng_scale = max(math.cos(math.radians(lat)), 0.01)
        lng_delta = math.degrees(angular / lng_scale)
        epsilon = 0.00001
        clauses = [
            "g.max_lng >= ?",
            "g.min_lng <= ?",
            "g.max_lat >= ?",
            "g.min_lat <= ?",
        ]
        params: List[Any] = [
            max(-180.0, lng - lng_delta - epsilon),
            min(180.0, lng + lng_delta + epsilon),
            max(-90.0, lat - lat_delta - epsilon),
            min(90.0, lat + lat_delta + epsilon),
        ]
        match = _fts_query(query)
        joins = ""
        if match:
            joins = " JOIN poi_fts ON poi_fts.rowid = p.id"
            clauses.append("poi_fts MATCH ?")
            params.append(match)
        else:
            text_sql, text_params = _fallback_text_sql(
                query, "p", ("name", "category", "category_path", "search_text")
            )
            clauses.append("(" + text_sql + ")")
            params.extend(text_params)
        filter_sql, filter_params = _filter_sql(filters or {})
        if filter_sql:
            clauses.append(filter_sql)
            params.extend(filter_params)
        columns = _columns("p", _POI_FIELDS)
        sql = (
            "SELECT %s FROM poi_geo g JOIN pois p ON p.id = g.id%s WHERE %s"
            % (columns, joins, " AND ".join(clauses))
        )
        query_terms = _query_tokens(query)
        heap: List[Tuple[Tuple[float, float, int], Dict[str, Any]]] = []
        with _connection(self.path, self.build_id) as connection:
            cursor = connection.execute(sql, params)
            while True:
                rows = cursor.fetchmany(2000)
                if not rows:
                    break
                for row in rows:
                    distance = haversine_m(lng, lat, row["longitude"], row["latitude"])
                    if distance > distance_m:
                        continue
                    lexical_score = _coverage_score(
                        query_terms, normalize_query(row["search_text"])
                    )
                    if query_terms and lexical_score <= 0:
                        continue
                    item = _poi_dict(row)
                    item["distance_m"] = round(distance, 1)
                    item["text_match_score"] = round(lexical_score, 6)
                    key = (lexical_score, -distance, -int(row["id"]))
                    heapq.heappush(heap, (key, item))
                    if len(heap) > top_k:
                        heapq.heappop(heap)
        return [item for _, item in sorted(heap, key=lambda pair: pair[0], reverse=True)]

    def nearby_row_distances(
        self,
        *,
        lng: float,
        lat: float,
        distance_m: float,
        filters: Optional[Mapping[str, Any]] = None,
    ) -> Dict[int, float]:
        if not all(math.isfinite(value) for value in (lng, lat, distance_m)):
            raise ValueError("经纬度和distance_m必须是有限数值")
        if not -180 <= lng <= 180 or not -90 <= lat <= 90 or distance_m <= 0:
            raise ValueError("经纬度越界或distance_m非正数")
        angular = distance_m / 6371008.8
        lat_delta = math.degrees(angular)
        lng_scale = max(math.cos(math.radians(lat)), 0.01)
        lng_delta = math.degrees(angular / lng_scale)
        epsilon = 0.00001
        clauses = [
            "g.max_lng >= ?",
            "g.min_lng <= ?",
            "g.max_lat >= ?",
            "g.min_lat <= ?",
        ]
        params: List[Any] = [
            max(-180.0, lng - lng_delta - epsilon),
            min(180.0, lng + lng_delta + epsilon),
            max(-90.0, lat - lat_delta - epsilon),
            min(90.0, lat + lat_delta + epsilon),
        ]
        filter_sql, filter_params = _filter_sql(filters or {})
        if filter_sql:
            clauses.append(filter_sql)
            params.extend(filter_params)
        sql = (
            "SELECT p.id, p.longitude, p.latitude "
            "FROM poi_geo g JOIN pois p ON p.id = g.id WHERE %s"
            % " AND ".join(clauses)
        )
        distances: Dict[int, float] = {}
        with _connection(self.path, self.build_id) as connection:
            cursor = connection.execute(sql, params)
            while True:
                rows = cursor.fetchmany(5000)
                if not rows:
                    break
                for row in rows:
                    distance = haversine_m(lng, lat, row["longitude"], row["latitude"])
                    if distance <= distance_m:
                        distances[int(row["id"])] = distance
        return distances


class NoteRepository:
    def __init__(self, path: Path, expected_build_id: Optional[str] = None):
        self.path = Path(path)
        self.build_id = _check_database(
            self.path, ("notes", "note_fts"), "笔记RAG库", expected_build_id
        )

    def search(
        self, query: str, top_k: int = 10, category: str = ""
    ) -> List[Dict[str, Any]]:
        top_k = _validate_top_k(top_k)
        columns = _columns("n", _NOTE_FIELDS)
        match = _fts_query(query)
        candidate_limit = min(max(top_k * 5, top_k), 500)
        category_sql = ""
        category_params: List[Any] = []
        if category:
            category_sql = " AND lower(n.category_path) LIKE ? ESCAPE '\\'"
            category_params.append(_like_pattern(category))
        with _connection(self.path, self.build_id) as connection:
            if match:
                rows = connection.execute(
                    """
                    SELECT %s, bm25(note_fts, 5.0, 3.0, 1.0) AS retrieval_rank
                    FROM note_fts JOIN notes n ON n.id = note_fts.rowid
                    WHERE note_fts MATCH ?%s
                    ORDER BY retrieval_rank, n.id
                    LIMIT ?
                    """
                    % (columns, category_sql),
                    [match] + category_params + [candidate_limit],
                ).fetchall()
            else:
                text_sql, text_params = _fallback_text_sql(
                    query, "n", ("content", "open_tag", "open_tag_term", "poi_name")
                )
                rows = connection.execute(
                    "SELECT %s, 0.0 AS retrieval_rank FROM notes n "
                    "WHERE (%s)%s ORDER BY n.id LIMIT ?"
                    % (columns, text_sql, category_sql),
                    text_params + category_params + [candidate_limit],
                ).fetchall()
        diversified = []
        per_poi_counts: Dict[str, int] = {}
        for row in rows:
            poiid = row["poiid"]
            if per_poi_counts.get(poiid, 0) >= 2:
                continue
            per_poi_counts[poiid] = per_poi_counts.get(poiid, 0) + 1
            diversified.append(row)
            if len(diversified) >= top_k:
                break
        return [_note_dict(row, max_content_chars=600) for row in diversified]

    def count_by_poiid(self, poiid: str) -> int:
        with _connection(self.path, self.build_id) as connection:
            return int(
                connection.execute(
                    "SELECT count(*) FROM notes WHERE poiid = ?", (poiid,)
                ).fetchone()[0]
            )

    def by_row_ids(
        self,
        row_ids: Sequence[int],
        category: str = "",
        max_content_chars: int = 600,
    ) -> Dict[int, Dict[str, Any]]:
        identifiers = [int(value) for value in row_ids]
        if not identifiers:
            return {}
        placeholders = ",".join("?" for _ in identifiers)
        category_sql = ""
        params: List[Any] = list(identifiers)
        if category:
            category_sql = " AND lower(n.category_path) LIKE ? ESCAPE '\\'"
            params.append(_like_pattern(category))
        columns = _columns("n", _NOTE_FIELDS)
        with _connection(self.path, self.build_id) as connection:
            rows = connection.execute(
                "SELECT %s FROM notes n WHERE n.id IN (%s)%s"
                % (columns, placeholders, category_sql),
                params,
            ).fetchall()
        return {
            int(row["id"]): _note_dict(row, max_content_chars=max_content_chars)
            for row in rows
        }

    def filtered_row_ids(self, category: str = "") -> List[int]:
        category_sql = ""
        params: List[Any] = []
        if category:
            category_sql = " WHERE lower(n.category_path) LIKE ? ESCAPE '\\'"
            params.append(_like_pattern(category))
        with _connection(self.path, self.build_id) as connection:
            rows = connection.execute(
                "SELECT n.id FROM notes n%s ORDER BY n.id" % category_sql,
                params,
            ).fetchall()
        return [int(row[0]) for row in rows]

    def by_poiid(self, poiid: str, limit: int = 10) -> List[Dict[str, Any]]:
        limit = _validate_top_k(limit)
        columns = _columns("n", _NOTE_FIELDS)
        with _connection(self.path, self.build_id) as connection:
            rows = connection.execute(
                """
                SELECT %s, 0.0 AS retrieval_rank FROM notes n
                WHERE poiid = ?
                ORDER BY COALESCE(hot_score, 0) DESC, created_at DESC, id
                LIMIT ?
                """
                % columns,
                (poiid, limit),
            ).fetchall()
        return [_note_dict(row, max_content_chars=1800) for row in rows]


def _note_dict(row: sqlite3.Row, *, max_content_chars: int) -> Dict[str, Any]:
    content = row["content"] or ""
    truncated = len(content) > max_content_chars
    if truncated:
        content = content[:max_content_chars].rstrip() + "…"
    item = {
        "nid": row["nid"],
        "poiid": row["poiid"],
        "poi_name": row["poi_name"],
        "content": content,
        "content_chars": len(row["content"] or ""),
        "content_truncated": truncated,
        "category_path": row["category_path"],
        "open_tag": row["open_tag"],
        "open_tag_term": row["open_tag_term"],
        "open_tag_detail": row["open_tag_detail"],
        "created_at": row["created_at"],
        "modified_at": row["modified_at"],
        "hot_score": row["hot_score"],
        "metrics": {
            "like": row["like_num"],
            "view": row["view_num"],
            "share": row["share_num"],
            "reply": row["reply_num"],
            "favorites": row["favorites_num"],
        },
    }
    if "retrieval_rank" in row.keys():
        item["retrieval_score"] = round(-float(row["retrieval_rank"]), 8)
    return item


def haversine_m(lng1: float, lat1: float, lng2: float, lat2: float) -> float:
    radius = 6371008.8
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)
    value = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * radius * math.asin(math.sqrt(value))


def _coverage_score(terms: Sequence[str], text: str) -> float:
    if not terms:
        return 1.0
    weights = [max(len(term) - 1, 1) for term in terms]
    matched = sum(weight for term, weight in zip(terms, weights) if term in text)
    return matched / float(sum(weights) or 1)
