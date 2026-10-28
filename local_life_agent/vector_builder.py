from __future__ import annotations

import gc
import hashlib
import json
import multiprocessing
import os
import sqlite3
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .dense_retrieval import (
    DEFAULT_DIMENSION,
    DEFAULT_MAX_SEQ_LENGTH,
    DEFAULT_MODEL_NAME,
    DEFAULT_QUERY_PROMPT_NAME,
    DEFAULT_RERANKER_MAX_LENGTH,
    DEFAULT_RERANKER_MODEL_NAME,
    VECTOR_SCHEMA_VERSION,
    SharedTextEmbedder,
    embedding_cache_dir,
    mlx_embedding_model_dir,
)
from .text_representation import note_retrieval_text, poi_retrieval_text


def _poi_text(row: sqlite3.Row) -> str:
    return poi_retrieval_text(dict(row))


def _note_text(row: sqlite3.Row) -> str:
    return note_retrieval_text(dict(row))


_WORKER_EMBEDDER: Optional[Any] = None
_WORKER_EMBEDDER_KEY: Optional[Tuple[Any, ...]] = None


def _worker_embedder(config: Dict[str, Any]) -> Any:
    """Return one lazily loaded model per worker process."""
    global _WORKER_EMBEDDER, _WORKER_EMBEDDER_KEY
    key = (
        config.get("backend", "torch"),
        config["model_name"],
        int(config["dimension"]),
        config["query_prompt_name"],
        int(config["max_seq_length"]),
        config["device"],
        config["cache_dir"],
        config.get("mlx_model_path", ""),
    )
    if _WORKER_EMBEDDER is None or _WORKER_EMBEDDER_KEY != key:
        if config.get("backend") == "mlx":
            from .dense_retrieval import SharedMlxTextEmbedder

            _WORKER_EMBEDDER = SharedMlxTextEmbedder(
                model_name=config["model_name"],
                model_path=Path(config["mlx_model_path"]),
                dimension=int(config["dimension"]),
                query_prompt_name=config["query_prompt_name"],
                max_seq_length=int(config["max_seq_length"]),
            )
        else:
            _WORKER_EMBEDDER = SharedTextEmbedder(
                model_name=config["model_name"],
                dimension=int(config["dimension"]),
                query_prompt_name=config["query_prompt_name"],
                max_seq_length=int(config["max_seq_length"]),
                cache_dir=Path(config["cache_dir"]),
                device=config["device"],
                local_files_only=True,
            )
        _WORKER_EMBEDDER_KEY = key
    return _WORKER_EMBEDDER


def _build_shard_worker(task: Dict[str, Any]) -> Dict[str, Any]:
    """Build one independently publishable FAISS shard."""
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("缺少faiss-cpu，请安装requirements-vector.txt") from exc
    embedder = _worker_embedder(task["embedder"])
    database = Path(task["database"])
    output_path = Path(task["output_path"])
    building_path = output_path.with_suffix(output_path.suffix + ".building")
    if building_path.exists():
        building_path.unlink()
    connection = sqlite3.connect(str(database))
    connection.row_factory = sqlite3.Row
    index = faiss.IndexIDMap2(faiss.IndexFlatIP(embedder.dimension))
    completed = 0
    started = time.time()
    try:
        cursor = connection.execute(
            task["select_sql"] + " WHERE id BETWEEN ? AND ? ORDER BY id",
            (int(task["first_id"]), int(task["last_id"])),
        )
        read_chunk_size = max(256, int(task["batch_size"]) * 32)
        while True:
            rows = cursor.fetchmany(read_chunk_size)
            if not rows:
                break
            builder = _poi_text if task["table"] == "pois" else _note_text
            # Group similar lengths so padding does not make every item as
            # expensive as the longest text in a batch. FAISS uses explicit
            # SQLite IDs, therefore insertion order has no semantic effect.
            text_rows = sorted(
                ((builder(row), int(row["id"])) for row in rows),
                key=lambda pair: len(pair[0]),
            )
            vectors = embedder.encode_documents(
                [pair[0] for pair in text_rows],
                batch_size=int(task["batch_size"]),
            )
            identifiers = np.asarray([pair[1] for pair in text_rows], dtype=np.int64)
            index.add_with_ids(vectors, identifiers)
            completed += len(rows)
    finally:
        connection.close()
    expected = int(task["expected_count"])
    if int(index.ntotal) != expected:
        raise RuntimeError(
            "%s分片%d向量数不一致: %d != %d"
            % (task["table"], task["shard_number"], index.ntotal, expected)
        )
    faiss.write_index(index, str(building_path))
    os.replace(building_path, output_path)
    elapsed = max(time.time() - started, 0.001)
    return {
        "shard_number": int(task["shard_number"]),
        "count": completed,
        "elapsed_seconds": round(elapsed, 3),
        "items_per_second": round(completed / elapsed, 3),
        "device": embedder.device,
    }


def _build_one(
    *,
    database: Path,
    table: str,
    select_sql: str,
    text_builder,
    output_path: Path,
    embedder: SharedTextEmbedder,
    batch_size: int,
    progress_every: int,
) -> int:
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("缺少faiss-cpu，请安装requirements-vector.txt") from exc
    connection = sqlite3.connect(str(database))
    connection.row_factory = sqlite3.Row
    total = int(connection.execute("SELECT count(*) FROM %s" % table).fetchone()[0])
    index = faiss.IndexIDMap2(faiss.IndexFlatIP(embedder.dimension))
    cursor = connection.execute(select_sql)
    completed = 0
    started = time.time()
    # Give SentenceTransformer a larger block so it can group similar lengths
    # internally, while keeping the actual accelerator batch deliberately small.
    read_chunk_size = max(256, batch_size * 32)
    try:
        while True:
            rows = cursor.fetchmany(read_chunk_size)
            if not rows:
                break
            texts = [text_builder(row) for row in rows]
            vectors = embedder.encode_documents(texts, batch_size=batch_size)
            identifiers = np.asarray([row["id"] for row in rows], dtype=np.int64)
            index.add_with_ids(vectors, identifiers)
            completed += len(rows)
            if progress_every and (
                completed == total or completed % progress_every < len(rows)
            ):
                elapsed = max(time.time() - started, 0.001)
                print(
                    "%s向量: %d/%d (%.1f 条/s)"
                    % (table, completed, total, completed / elapsed),
                    flush=True,
                )
    finally:
        connection.close()
    if int(index.ntotal) != total:
        raise RuntimeError("%s向量数不一致: %d != %d" % (table, index.ntotal, total))
    faiss.write_index(index, str(output_path))
    del index
    gc.collect()
    return total


def _table_bounds_and_shards(
    database: Path,
    table: str,
    shard_count: int,
) -> Tuple[int, List[Dict[str, int]]]:
    connection = sqlite3.connect(str(database))
    try:
        count, minimum, maximum = connection.execute(
            "SELECT count(*), min(id), max(id) FROM %s" % table
        ).fetchone()
        total = int(count)
        if total == 0:
            return 0, []
        minimum = int(minimum)
        maximum = int(maximum)
        width = max(1, (maximum - minimum + 1 + shard_count - 1) // shard_count)
        shards: List[Dict[str, int]] = []
        first_id = minimum
        shard_number = 0
        while first_id <= maximum:
            last_id = min(maximum, first_id + width - 1)
            expected = int(
                connection.execute(
                    "SELECT count(*) FROM %s WHERE id BETWEEN ? AND ?" % table,
                    (first_id, last_id),
                ).fetchone()[0]
            )
            if expected:
                shards.append(
                    {
                        "shard_number": shard_number,
                        "first_id": first_id,
                        "last_id": last_id,
                        "expected_count": expected,
                    }
                )
                shard_number += 1
            first_id = last_id + 1
        return total, shards
    finally:
        connection.close()


def _valid_shard(path: Path, *, dimension: int, expected_count: int) -> bool:
    if not path.is_file():
        return False
    try:
        import faiss

        index = faiss.read_index(str(path))
        return int(index.d) == dimension and int(index.ntotal) == expected_count
    except Exception:
        return False


def _merge_shards(
    shard_paths: Sequence[Path],
    *,
    dimension: int,
    expected_count: int,
    output_path: Path,
) -> None:
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("缺少faiss-cpu，请安装requirements-vector.txt") from exc
    merged = faiss.IndexIDMap2(faiss.IndexFlatIP(dimension))
    for path in shard_paths:
        shard = faiss.read_index(str(path))
        count = int(shard.ntotal)
        identifiers = faiss.vector_to_array(shard.id_map).astype(np.int64, copy=False)
        vectors = np.empty((count, dimension), dtype=np.float32)
        shard.index.reconstruct_n(0, count, vectors)
        merged.add_with_ids(vectors, identifiers)
        del shard, identifiers, vectors
        gc.collect()
    if int(merged.ntotal) != expected_count:
        raise RuntimeError(
            "合并后的向量数不一致: %d != %d" % (merged.ntotal, expected_count)
        )
    faiss.write_index(merged, str(output_path))


def _build_one_resumable(
    *,
    database: Path,
    table: str,
    select_sql: str,
    output_path: Path,
    work_dir: Path,
    embedder_config: Dict[str, Any],
    batch_size: int,
    workers: int,
    shard_count: int,
) -> Tuple[int, str]:
    total, shards = _table_bounds_and_shards(database, table, shard_count)
    table_dir = work_dir / table
    table_dir.mkdir(parents=True, exist_ok=True)
    tasks: List[Dict[str, Any]] = []
    shard_paths: List[Path] = []
    reused = 0
    for shard in shards:
        path = table_dir / ("part-%04d.faiss" % shard["shard_number"])
        shard_paths.append(path)
        if _valid_shard(
            path,
            dimension=int(embedder_config["dimension"]),
            expected_count=shard["expected_count"],
        ):
            reused += shard["expected_count"]
            continue
        task: Dict[str, Any] = dict(shard)
        task.update(
            {
                "database": str(database),
                "table": table,
                "select_sql": select_sql,
                "output_path": str(path),
                "embedder": embedder_config,
                "batch_size": batch_size,
            }
        )
        tasks.append(task)
    print(
        "%s向量分片: 总计%d，已复用%d，待构建%d（%d workers）"
        % (table, total, reused, total - reused, workers),
        flush=True,
    )
    completed = reused
    device_used = str(embedder_config["device"])
    if tasks:
        if workers == 1:
            results = (_build_shard_worker(task) for task in tasks)
            for result in results:
                completed += int(result["count"])
                device_used = str(result["device"])
                print(
                    "%s向量: %d/%d；分片%04d %.1f条/s"
                    % (
                        table,
                        completed,
                        total,
                        result["shard_number"],
                        result["items_per_second"],
                    ),
                    flush=True,
                )
        else:
            if embedder_config["device"] == "mps":
                raise ValueError("MPS构建必须使用--workers 1；多进程请使用CPU")
            context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
                futures = {
                    executor.submit(_build_shard_worker, task): task for task in tasks
                }
                for future in as_completed(futures):
                    result = future.result()
                    completed += int(result["count"])
                    device_used = str(result["device"])
                    print(
                        "%s向量: %d/%d；分片%04d %.1f条/s"
                        % (
                            table,
                            completed,
                            total,
                            result["shard_number"],
                            result["items_per_second"],
                        ),
                        flush=True,
                    )
    merge_path = output_path.with_suffix(output_path.suffix + ".merging")
    if merge_path.exists():
        merge_path.unlink()
    _merge_shards(
        shard_paths,
        dimension=int(embedder_config["dimension"]),
        expected_count=total,
        output_path=merge_path,
    )
    os.replace(merge_path, output_path)
    return total, device_used


def build_vector_indexes(
    index_dir: Path,
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    dimension: int = DEFAULT_DIMENSION,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    query_prompt_name: str = DEFAULT_QUERY_PROMPT_NAME,
    reranker_model_name: str = DEFAULT_RERANKER_MODEL_NAME,
    reranker_max_length: int = DEFAULT_RERANKER_MAX_LENGTH,
    embedding_backend: str = "torch",
    mlx_model_path: Optional[Path] = None,
    device: str = "auto",
    poi_batch_size: int = 16,
    note_batch_size: int = 8,
    workers: int = 1,
    shard_count: int = 64,
    progress_every: int = 5000,
    overwrite: bool = False,
) -> Dict[str, Any]:
    if (
        dimension < 1
        or poi_batch_size < 1
        or note_batch_size < 1
        or workers < 1
        or shard_count < 1
        or max_seq_length < 8
        or reranker_max_length < 8
    ):
        raise ValueError("dimension、batch size和模型最大长度必须为正数")
    if embedding_backend not in {"torch", "mlx"}:
        raise ValueError("embedding_backend只能是torch或mlx")
    if embedding_backend == "mlx" and workers != 1:
        raise ValueError("MLX构建必须使用--workers 1")
    index_dir = Path(index_dir).resolve()
    lexical_manifest_path = index_dir / "manifest.json"
    poi_db = index_dir / "poi_rag.sqlite3"
    note_db = index_dir / "note_rag.sqlite3"
    for path in (lexical_manifest_path, poi_db, note_db):
        if not path.is_file():
            raise FileNotFoundError("基础索引不存在: %s" % path)
    lexical_manifest = json.loads(lexical_manifest_path.read_text(encoding="utf-8"))
    poi_output = index_dir / "poi_vectors.faiss"
    note_output = index_dir / "note_vectors.faiss"
    manifest_output = index_dir / "vector_manifest.json"
    existing = [p for p in (poi_output, note_output, manifest_output) if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "向量索引已存在；如需重建请传--overwrite: "
            + ", ".join(str(path) for path in existing)
        )
    poi_building = index_dir / "poi_vectors.faiss.building"
    note_building = index_dir / "note_vectors.faiss.building"
    manifest_building = index_dir / "vector_manifest.json.building"
    for path in (poi_building, note_building, manifest_building):
        if path.exists():
            path.unlink()

    started = time.time()
    build_signature = {
        "source_build_id": lexical_manifest.get("build_id"),
        "model_name": model_name,
        "dimension": dimension,
        "query_prompt_name": query_prompt_name,
        "max_seq_length": max_seq_length,
        "embedding_backend": embedding_backend,
        "device": device,
        "shard_count": shard_count,
    }
    signature = hashlib.sha256(
        json.dumps(build_signature, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    work_dir = index_dir / ".vector_build_parts" / signature
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "build_config.json").write_text(
        json.dumps(build_signature, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    embedder_config = {
        "backend": embedding_backend,
        "model_name": model_name,
        "dimension": dimension,
        "query_prompt_name": query_prompt_name,
        "max_seq_length": max_seq_length,
        "cache_dir": str(embedding_cache_dir()),
        "device": device,
        "mlx_model_path": str(
            Path(mlx_model_path).resolve()
            if mlx_model_path is not None
            else mlx_embedding_model_dir().resolve()
        ),
    }
    poi_count, poi_device = _build_one_resumable(
        database=poi_db,
        table="pois",
        select_sql=(
            "SELECT id,name,category,category_path,address,district,business_area,tags "
            "FROM pois"
        ),
        output_path=poi_building,
        work_dir=work_dir,
        embedder_config=embedder_config,
        batch_size=poi_batch_size,
        workers=workers,
        shard_count=shard_count,
    )
    note_count, note_device = _build_one_resumable(
        database=note_db,
        table="notes",
        select_sql=(
            "SELECT id,poiid,poi_name,content,category_path,open_tag "
            "FROM notes"
        ),
        output_path=note_building,
        work_dir=work_dir,
        embedder_config=embedder_config,
        batch_size=note_batch_size,
        workers=workers,
        shard_count=shard_count,
    )
    manifest = {
        "vector_schema_version": VECTOR_SCHEMA_VERSION,
        "vector_build_id": uuid.uuid4().hex,
        "source_build_id": lexical_manifest.get("build_id"),
        "model_name": model_name,
        "embedding_backend": embedding_backend,
        "mlx_model_dir_name": (
            Path(embedder_config["mlx_model_path"]).name
            if embedding_backend == "mlx"
            else None
        ),
        "dimension": dimension,
        "query_prompt_name": query_prompt_name,
        "max_seq_length": max_seq_length,
        "poi_batch_size_used_for_build": poi_batch_size,
        "note_batch_size_used_for_build": note_batch_size,
        "workers_used_for_build": workers,
        "shard_count_used_for_build": shard_count,
        "metric": "cosine_via_normalized_inner_product",
        "index_type": "IndexIDMap2(IndexFlatIP)",
        "reranker": {
            "model_name": reranker_model_name,
            "max_length": reranker_max_length,
            "score_activation": "identity_logit_difference",
            "instruction": "official_default_query_prompt",
        },
        "poi_count": poi_count,
        "note_count": note_count,
        "device_used_for_build": (
            poi_device if poi_device == note_device else [poi_device, note_device]
        ),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    manifest_building.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(poi_building, poi_output)
    os.replace(note_building, note_output)
    os.replace(manifest_building, manifest_output)
    return manifest
