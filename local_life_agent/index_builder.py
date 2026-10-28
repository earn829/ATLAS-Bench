from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple, Union


try:
    import fcntl
except ImportError:  # pragma: no cover - Linux/macOS benchmark environment has it
    fcntl = None


SCHEMA_VERSION = 3
TOKENIZER_VERSION = "cjk_bigram_latin_word_v1"
_SEGMENT_RE = re.compile(r"[\u3400-\u9fff]+|[a-zA-Z0-9]+")


def iter_json_array(path: Path, chunk_size: int = 1024 * 1024) -> Iterator[Dict[str, Any]]:
    """Stream a large top-level JSON array without loading it into memory."""

    decoder = json.JSONDecoder()
    buffer = ""
    started = False
    finished = False
    need_separator = False
    after_comma = False
    with Path(path).open(encoding="utf-8") as handle:
        while not finished:
            chunk = handle.read(chunk_size)
            eof = not chunk
            buffer += chunk
            position = 0
            length = len(buffer)
            if not started:
                while position < length and buffer[position].isspace():
                    position += 1
                if position >= length:
                    if eof:
                        raise ValueError("JSON 文件为空")
                    buffer = buffer[position:]
                    continue
                if buffer[position] != "[":
                    raise ValueError("源文件顶层必须是 JSON array")
                started = True
                position += 1
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position >= len(buffer):
                    break
                if need_separator:
                    if buffer[position] == ",":
                        position += 1
                        need_separator = False
                        after_comma = True
                        continue
                    if buffer[position] != "]":
                        raise ValueError("JSON 数组元素之间缺少逗号")
                if buffer[position] == "]":
                    if after_comma:
                        raise ValueError("JSON 数组不允许尾随逗号")
                    finished = True
                    position += 1
                    break
                try:
                    value, end = decoder.raw_decode(buffer, position)
                except json.JSONDecodeError:
                    break
                if not isinstance(value, dict):
                    raise ValueError("源数组元素必须是 JSON object")
                yield value
                position = end
                need_separator = True
                after_comma = False
            buffer = buffer[position:]
            if eof and not finished:
                if buffer.strip():
                    raise ValueError("JSON 文件在对象中间结束")
                raise ValueError("JSON 文件缺少数组结束符 ]")
        buffer += handle.read()
        if buffer.strip():
            raise ValueError("JSON 数组结束符后存在多余内容")


def iter_json_arrays(paths: Sequence[Path]) -> Iterator[Dict[str, Any]]:
    for path in paths:
        yield from iter_json_array(path)


def _mapping(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return " ".join(_text(item) for item in value if item is not None)
    return str(value).strip()


def _float(value: Any) -> Optional[float]:
    try:
        parsed = float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None
    return parsed if parsed is not None and math.isfinite(parsed) else None


def _int(value: Any) -> Optional[int]:
    parsed = _float(value)
    return int(parsed) if parsed is not None else None


def _district(pcdr: str) -> str:
    candidates = [
        part.strip()
        for part in pcdr.split(";")
        if part.strip().endswith(("区", "县"))
        and not part.strip().endswith("自治区")
    ]
    return candidates[0] if candidates else ""


def _lexical_tokens(text: str) -> str:
    """Pre-tokenize Chinese as bigrams so two-character constraints are searchable."""

    tokens = []
    seen = set()
    for segment in _SEGMENT_RE.findall(_text(text).lower()):
        is_cjk = "\u3400" <= segment[0] <= "\u9fff"
        if is_cjk:
            if len(segment) <= 2:
                candidates = [segment]
            else:
                candidates = [segment[index : index + 2] for index in range(len(segment) - 1)]
                if len(segment) <= 16:
                    candidates.append(segment)
        else:
            candidates = [segment]
            if len(segment) > 4:
                candidates.extend(
                    segment[index : index + 3] for index in range(len(segment) - 2)
                )
        for token in candidates:
            if token and token not in seen:
                seen.add(token)
                tokens.append(token)
    return " ".join(tokens)


def extract_note_body(raw_content: str) -> str:
    """Remove the generated metadata wrapper while preserving the note body."""

    marker = "笔记正文："
    if marker in raw_content:
        body = raw_content.split(marker, 1)[1].strip()
        if body:
            return body
    return raw_content.strip()


def _open_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(str(path))
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute("PRAGMA cache_size=-200000")
    connection.execute("PRAGMA user_version=%d" % SCHEMA_VERSION)
    return connection


def _create_poi_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE pois (
            id INTEGER PRIMARY KEY,
            poiid TEXT NOT NULL UNIQUE,
            name TEXT NOT NULL,
            latitude REAL,
            longitude REAL,
            address TEXT,
            pcdr TEXT,
            district TEXT,
            business_area TEXT,
            category TEXT,
            category_path TEXT,
            rating REAL,
            average_cost REAL,
            star REAL,
            open_time TEXT,
            tags TEXT,
            tickets TEXT,
            mall_hotel_hours_flag TEXT,
            parking_spaces INTEGER,
            comment INTEGER NOT NULL,
            comment_count INTEGER NOT NULL,
            search_text TEXT NOT NULL,
            payload_json TEXT NOT NULL
        );
        CREATE VIRTUAL TABLE poi_geo USING rtree(
            id,
            min_lng, max_lng,
            min_lat, max_lat
        );
        CREATE VIRTUAL TABLE poi_fts USING fts5(
            name, category, search_text,
            content='', tokenize='unicode61 remove_diacritics 2'
        );
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """
    )


def _create_note_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE notes (
            id INTEGER PRIMARY KEY,
            nid TEXT NOT NULL UNIQUE,
            poiid TEXT NOT NULL,
            poi_name TEXT NOT NULL,
            content TEXT NOT NULL,
            related_items TEXT,
            location TEXT,
            created_at TEXT,
            modified_at TEXT,
            category_path TEXT,
            open_tag TEXT,
            open_tag_term TEXT,
            open_tag_detail TEXT,
            hot_score REAL,
            like_num INTEGER,
            view_num INTEGER,
            share_num INTEGER,
            reply_num INTEGER,
            favorites_num INTEGER,
            payload_json TEXT NOT NULL,
            UNIQUE(poiid, nid)
        );
        CREATE INDEX idx_notes_poiid ON notes(poiid);
        CREATE VIRTUAL TABLE note_fts USING fts5(
            poi_name, open_tag, search_text,
            content='', tokenize='unicode61 remove_diacritics 2'
        );
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """
    )


def _finalize(
    poi_connection: sqlite3.Connection, note_connection: sqlite3.Connection
) -> None:
    poi_connection.executescript(
        """
        CREATE INDEX idx_pois_name ON pois(name);
        CREATE INDEX idx_pois_category ON pois(category);
        CREATE INDEX idx_pois_district ON pois(district);
        CREATE INDEX idx_pois_business_area ON pois(business_area);
        CREATE INDEX idx_pois_rating ON pois(rating);
        CREATE INDEX idx_pois_average_cost ON pois(average_cost);
        CREATE INDEX idx_pois_comment ON pois(comment);
        INSERT INTO poi_fts(poi_fts) VALUES('optimize');
        ANALYZE;
        """
    )
    note_connection.executescript(
        """
        CREATE INDEX idx_notes_created_at ON notes(created_at);
        INSERT INTO note_fts(note_fts) VALUES('optimize');
        ANALYZE;
        """
    )


def _remove_sqlite_family(path: Path, *, include_main: bool = True) -> None:
    suffixes = ("", "-wal", "-shm") if include_main else ("-wal", "-shm")
    for suffix in suffixes:
        target = Path(str(path) + suffix)
        if target.exists():
            target.unlink()


def _validate_database(
    connection: sqlite3.Connection, table: str, expected_count: int
) -> None:
    integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    if integrity != "ok":
        raise RuntimeError("%s 数据库完整性检查失败: %s" % (table, integrity))
    actual = connection.execute("SELECT count(*) FROM %s" % table).fetchone()[0]
    if actual != expected_count:
        raise RuntimeError(
            "%s 条数校验失败: expected=%d actual=%d"
            % (table, expected_count, actual)
        )
    version = connection.execute("PRAGMA user_version").fetchone()[0]
    if version != SCHEMA_VERSION:
        raise RuntimeError("数据库schema版本错误: %s" % version)


def _build_indexes_unlocked(
    source: Union[Path, Sequence[Path]],
    output_dir: Path,
    *,
    overwrite: bool = False,
    limit: Optional[int] = None,
    progress_every: int = 10000,
) -> Dict[str, Any]:
    """Split source data into atomically published POI and note retrieval DBs."""

    if limit is not None and limit < 1:
        raise ValueError("limit 必须大于0")
    if progress_every < 0:
        raise ValueError("progress_every 不能为负")
    raw_sources = [source] if isinstance(source, Path) else list(source)
    if not raw_sources:
        raise ValueError("至少需要一个源数据文件")
    source_paths: Tuple[Path, ...] = tuple(Path(path).resolve() for path in raw_sources)
    for source_path in source_paths:
        if not source_path.is_file():
            raise FileNotFoundError("源数据不存在: %s" % source_path)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    poi_path = output_dir / "poi_rag.sqlite3"
    note_path = output_dir / "note_rag.sqlite3"
    manifest_path = output_dir / "manifest.json"
    poi_building = output_dir / "poi_rag.sqlite3.building"
    note_building = output_dir / "note_rag.sqlite3.building"
    manifest_building = output_dir / "manifest.json.building"
    existing = [path for path in (poi_path, note_path, manifest_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "索引已存在；如需重建请传 --overwrite: "
            + ", ".join(str(path) for path in existing)
        )
    for temporary in (poi_building, note_building):
        _remove_sqlite_family(temporary)
    if manifest_building.exists():
        manifest_building.unlink()

    started = time.time()
    build_id = uuid.uuid4().hex
    source_stats = tuple(path.stat() for path in source_paths)
    poi_db: Optional[sqlite3.Connection] = None
    note_db: Optional[sqlite3.Connection] = None
    poi_count = 0
    note_count = 0
    source_record_count = 0
    missing_note_id_records = 0
    duplicate_note_records = 0
    empty_note_records = 0
    invalid_coordinate_records = 0
    invalid_poi_records = 0
    try:
        poi_db = _open_database(poi_building)
        note_db = _open_database(note_building)
        _create_poi_schema(poi_db)
        _create_note_schema(note_db)
        for connection, database_kind in (
            (poi_db, "poi"),
            (note_db, "note"),
        ):
            connection.executemany(
                "INSERT INTO metadata(key,value) VALUES(?,?)",
                (
                    ("build_id", build_id),
                    ("database_kind", database_kind),
                    ("schema_version", str(SCHEMA_VERSION)),
                    ("tokenizer_version", TOKENIZER_VERSION),
                ),
            )

        for record in iter_json_arrays(source_paths):
            if limit is not None and poi_count >= limit:
                break
            source_record_count += 1
            poiid = _text(record.get("poiid"))
            name = _text(record.get("name"))
            if not poiid or not name:
                invalid_poi_records += 1
                continue
            feature = _mapping(record.get("poi_feature"))
            basic = _mapping(feature.get("basic_info"))
            service = _mapping(feature.get("service_info"))
            comments_value = record.get("comment_features")
            comments = comments_value if isinstance(comments_value, list) else []

            latitude = _float(basic.get("latitude"))
            longitude = _float(basic.get("longitude"))
            if (
                latitude is None
                or longitude is None
                or not -90 <= latitude <= 90
                or not -180 <= longitude <= 180
            ):
                if latitude is not None or longitude is not None:
                    invalid_coordinate_records += 1
                latitude = None
                longitude = None
            address = _text(basic.get("address"))
            pcdr = _text(basic.get("pcdr"))
            category = _text(basic.get("tag_display_std"))
            category_path = _text(service.get("t_tag_class"))
            business_area = _text(service.get("business_area"))
            open_time = _text(service.get("opentime_disp"))
            tickets = _text(service.get("tickets_des"))
            tags = " ".join(
                part
                for part in (
                    _text(service.get("atag")),
                    _text(service.get("open_tag_forward")),
                    tickets,
                )
                if part
            )
            search_text = " ".join(
                part
                for part in (
                    poiid,
                    name,
                    category,
                    category_path,
                    address,
                    pcdr,
                    business_area,
                    open_time,
                    tags,
                    _text(service.get("star")),
                )
                if part
            )

            persisted_for_poi = 0
            for comment in comments:
                if not isinstance(comment, dict):
                    empty_note_records += 1
                    continue
                comment_basic = _mapping(comment.get("basic_info"))
                raw_content = _text(comment_basic.get("content"))
                if not raw_content:
                    empty_note_records += 1
                    continue
                nid = _text(comment_basic.get("nid"))
                if not nid:
                    missing_note_id_records += 1
                    continue
                details = _mapping(comment.get("details"))
                content = extract_note_body(raw_content)
                open_tag = _text(details.get("open_tag"))
                open_tag_term = _text(details.get("open_tag_term"))
                open_tag_detail = _text(details.get("open_tag_detail"))
                note_search_text = " ".join(
                    part
                    for part in (
                        name,
                        category,
                        category_path,
                        open_tag,
                        open_tag_term,
                        open_tag_detail,
                        content,
                    )
                    if part
                )
                try:
                    note_cursor = note_db.execute(
                        """
                        INSERT INTO notes(
                            nid,poiid,poi_name,content,related_items,location,created_at,
                            modified_at,category_path,open_tag,open_tag_term,open_tag_detail,
                            hot_score,like_num,view_num,share_num,reply_num,favorites_num,
                            payload_json
                        ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                        """,
                        (
                            nid,
                            poiid,
                            name,
                            content,
                            _text(comment_basic.get("related_items")),
                            _text(comment_basic.get("loc")),
                            _text(comment_basic.get("gmt_create")),
                            _text(comment_basic.get("gmt_modified")),
                            _text(comment_basic.get("t_tag2")) or category_path,
                            open_tag,
                            open_tag_term,
                            open_tag_detail,
                            _float(details.get("hot_score")),
                            _int(details.get("like_num")),
                            _int(details.get("view_num")),
                            _int(details.get("share_num")),
                            _int(details.get("reply_num")),
                            _int(details.get("favorites_num")),
                            json.dumps(comment, ensure_ascii=False, separators=(",", ":")),
                        ),
                    )
                except sqlite3.IntegrityError:
                    duplicate_note_records += 1
                    continue
                note_db.execute(
                    "INSERT INTO note_fts(rowid,poi_name,open_tag,search_text) VALUES(?,?,?,?)",
                    (
                        note_cursor.lastrowid,
                        _lexical_tokens(name),
                        _lexical_tokens(" ".join((open_tag, open_tag_term, open_tag_detail))),
                        _lexical_tokens(note_search_text),
                    ),
                )
                note_count += 1
                persisted_for_poi += 1

            poi_payload = {"poiid": poiid, "name": name, "poi_feature": feature}
            poi_cursor = poi_db.execute(
                """
                INSERT INTO pois(
                    poiid,name,latitude,longitude,address,pcdr,district,business_area,
                    category,category_path,rating,average_cost,star,open_time,tags,tickets,
                    mall_hotel_hours_flag,parking_spaces,comment,comment_count,search_text,
                    payload_json
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    poiid,
                    name,
                    latitude,
                    longitude,
                    address,
                    pcdr,
                    _district(pcdr),
                    business_area,
                    category,
                    category_path,
                    _float(service.get("rating")),
                    _float(service.get("averagecost")),
                    _float(service.get("star")),
                    open_time,
                    tags,
                    tickets,
                    _text(service.get("mall_hotel_hours_flag")),
                    _int(service.get("num_space_f")),
                    int(persisted_for_poi > 0),
                    persisted_for_poi,
                    search_text,
                    json.dumps(poi_payload, ensure_ascii=False, separators=(",", ":")),
                ),
            )
            rowid = poi_cursor.lastrowid
            poi_db.execute(
                "INSERT INTO poi_fts(rowid,name,category,search_text) VALUES(?,?,?,?)",
                (
                    rowid,
                    _lexical_tokens(name),
                    _lexical_tokens(" ".join((category, category_path))),
                    _lexical_tokens(search_text),
                ),
            )
            if latitude is not None and longitude is not None:
                poi_db.execute(
                    "INSERT INTO poi_geo VALUES(?,?,?,?,?)",
                    (rowid, longitude, longitude, latitude, latitude),
                )
            poi_count += 1
            if poi_count % 2000 == 0:
                poi_db.commit()
                note_db.commit()
            if progress_every and poi_count % progress_every == 0:
                elapsed = max(time.time() - started, 0.001)
                print(
                    "已处理 %d POI / %d 笔记 (%.1f POI/s)"
                    % (poi_count, note_count, poi_count / elapsed),
                    flush=True,
                )

        poi_db.commit()
        note_db.commit()
        _finalize(poi_db, note_db)
        poi_db.commit()
        note_db.commit()
        _validate_database(poi_db, "pois", poi_count)
        _validate_database(note_db, "notes", note_count)
        poi_db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        note_db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        poi_db.execute("PRAGMA journal_mode=DELETE")
        note_db.execute("PRAGMA journal_mode=DELETE")
        poi_db.close()
        note_db.close()
        poi_db = None
        note_db = None
        _remove_sqlite_family(poi_building, include_main=False)
        _remove_sqlite_family(note_building, include_main=False)

        source_entries = [
            {
                "path": str(path),
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "fingerprint": "%d:%d" % (stat.st_size, stat.st_mtime_ns),
            }
            for path, stat in zip(source_paths, source_stats)
        ]
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "build_id": build_id,
            "tokenizer_version": TOKENIZER_VERSION,
            "source": (
                str(source_paths[0])
                if len(source_paths) == 1
                else [str(path) for path in source_paths]
            ),
            "sources": source_entries,
            "source_size_bytes": sum(stat.st_size for stat in source_stats),
            "source_mtime_ns": max(stat.st_mtime_ns for stat in source_stats),
            "source_fingerprint": "|".join(
                "%s:%d:%d" % (path.name, stat.st_size, stat.st_mtime_ns)
                for path, stat in zip(source_paths, source_stats)
            ),
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "source_records_read": source_record_count,
            "poi_count": poi_count,
            "note_count": note_count,
            "missing_note_id_records": missing_note_id_records,
            "duplicate_note_records": duplicate_note_records,
            "empty_note_records": empty_note_records,
            "invalid_coordinate_records": invalid_coordinate_records,
            "invalid_poi_records": invalid_poi_records,
            "poi_database": str(poi_path),
            "note_database": str(note_path),
            "limit": limit,
            "elapsed_seconds": round(time.time() - started, 3),
        }
        with manifest_building.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
        if overwrite:
            _remove_sqlite_family(poi_path, include_main=False)
            _remove_sqlite_family(note_path, include_main=False)
        os.replace(poi_building, poi_path)
        os.replace(note_building, note_path)
        os.replace(manifest_building, manifest_path)
        return manifest
    except Exception:
        if poi_db is not None:
            poi_db.close()
        if note_db is not None:
            note_db.close()
        for temporary in (poi_building, note_building):
            _remove_sqlite_family(temporary)
        if manifest_building.exists():
            manifest_building.unlink()
        raise


def build_indexes(
    source: Union[Path, Sequence[Path]],
    output_dir: Path,
    *,
    overwrite: bool = False,
    limit: Optional[int] = None,
    progress_every: int = 10000,
) -> Dict[str, Any]:
    """Build under an exclusive directory lock and publish a matched DB pair."""

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / "build.lock"
    lock_handle = lock_path.open("a+", encoding="utf-8")
    if fcntl is None:
        lock_handle.close()
        raise RuntimeError("当前平台不支持fcntl，无法保证索引构建互斥")
    try:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("已有索引构建进程占用: %s" % lock_path) from exc
        lock_handle.seek(0)
        lock_handle.truncate()
        lock_handle.write("pid=%d started_at=%s\n" % (os.getpid(), time.time()))
        lock_handle.flush()
        return _build_indexes_unlocked(
            source,
            output_dir,
            overwrite=overwrite,
            limit=limit,
            progress_every=progress_every,
        )
    finally:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        finally:
            lock_handle.close()
