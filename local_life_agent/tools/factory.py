from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

from ..config import AgentConfig, PathConfig
from ..index_builder import SCHEMA_VERSION
from ..retrieval import NoteRepository, PoiRepository
from .nearby import SearchNearbyTool
from .note_search import SearchNotesTool
from .poi_notes import GetPoiNotesTool
from .poi_search import SearchPoisTool
from .registry import ToolRegistry


def build_default_registry(
    paths: PathConfig = PathConfig(), config: AgentConfig = AgentConfig()
) -> ToolRegistry:
    poi_parent = Path(paths.poi_db).resolve().parent
    note_parent = Path(paths.note_db).resolve().parent
    if poi_parent != note_parent:
        raise RuntimeError("POI库和笔记库必须来自同一个索引发布目录")
    manifest_path = poi_parent / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("索引manifest不存在: %s" % manifest_path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("索引manifest无法读取: %s" % manifest_path) from exc
    build_id = manifest.get("build_id")
    if not isinstance(build_id, str) or not build_id:
        raise RuntimeError("索引manifest缺少build_id；请重建索引")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(
            "manifest schema版本=%s，运行时要求=%s；请重建索引"
            % (manifest.get("schema_version"), SCHEMA_VERSION)
        )
    poi_count = manifest.get("poi_count")
    note_count = manifest.get("note_count")
    if not isinstance(poi_count, int) or not isinstance(note_count, int):
        raise RuntimeError("索引manifest缺少合法的poi_count或note_count")

    poi_repository = PoiRepository(paths.poi_db, expected_build_id=build_id)
    note_repository = NoteRepository(paths.note_db, expected_build_id=build_id)
    if config.enable_vector_recall:
        try:
            from ..dense_retrieval import (
                FaissVectorIndex,
                SharedMlxTextEmbedder,
                SharedTextEmbedder,
                embedding_cache_dir,
                load_vector_manifest,
                mlx_embedding_model_dir,
            )
            from ..neural_retrieval import (
                VectorRerankNoteRepository,
                VectorRerankPoiRepository,
            )
            from ..reranking import SharedQwen3Reranker

            vector_manifest = load_vector_manifest(
                poi_parent,
                expected_source_build_id=build_id,
                expected_poi_count=poi_count,
                expected_note_count=note_count,
            )
            device = os.environ.get("LOCAL_LIFE_VECTOR_DEVICE", "").strip()
            device = device or config.vector_device
            if device not in {"auto", "cpu", "cuda", "mps"}:
                raise ValueError(
                    "LOCAL_LIFE_VECTOR_DEVICE只能是auto、cpu、cuda或mps"
                )
            if vector_manifest["embedding_backend"] == "mlx":
                model_dir_name = str(vector_manifest.get("mlx_model_dir_name") or "")
                if not model_dir_name:
                    raise RuntimeError("MLX向量manifest缺少mlx_model_dir_name")
                embedder = SharedMlxTextEmbedder(
                    model_name=str(vector_manifest["model_name"]),
                    model_path=mlx_embedding_model_dir(model_dir_name),
                    dimension=int(vector_manifest["dimension"]),
                    query_prompt_name=str(vector_manifest["query_prompt_name"]),
                    max_seq_length=int(vector_manifest["max_seq_length"]),
                )
            else:
                embedder = SharedTextEmbedder(
                    model_name=str(vector_manifest["model_name"]),
                    dimension=int(vector_manifest["dimension"]),
                    query_prompt_name=str(vector_manifest["query_prompt_name"]),
                    max_seq_length=int(vector_manifest["max_seq_length"]),
                    cache_dir=embedding_cache_dir(),
                    device=device,
                    local_files_only=True,
                )
            poi_vector = FaissVectorIndex(
                poi_parent / "poi_vectors.faiss",
                embedder=embedder,
                expected_count=poi_count,
                label="POI向量索引",
            )
            note_vector = FaissVectorIndex(
                poi_parent / "note_vectors.faiss",
                embedder=embedder,
                expected_count=note_count,
                label="笔记向量索引",
            )
            reranker_manifest = vector_manifest["reranker"]
            reranker_device = os.environ.get(
                "LOCAL_LIFE_RERANKER_DEVICE", ""
            ).strip()
            reranker_device = reranker_device or device
            if reranker_device not in {"auto", "cpu", "cuda", "mps"}:
                raise ValueError(
                    "LOCAL_LIFE_RERANKER_DEVICE只能是auto、cpu、cuda或mps"
                )
            reranker = SharedQwen3Reranker(
                model_name=str(reranker_manifest["model_name"]),
                max_length=int(reranker_manifest["max_length"]),
                batch_size=config.reranker_batch_size,
                cache_dir=embedding_cache_dir(),
                device=reranker_device,
                local_files_only=True,
            )
            poi_repository = VectorRerankPoiRepository(
                poi_repository,
                poi_vector,
                reranker,
                candidate_k=config.vector_candidate_k,
            )
            note_repository = VectorRerankNoteRepository(
                note_repository,
                note_vector,
                reranker,
                candidate_k=config.vector_candidate_k,
            )
        except Exception as exc:
            if config.vector_required:
                raise RuntimeError(
                    "v5 Qwen向量召回与重排初始化失败；请安装requirements-vector.txt并重建向量索引"
                ) from exc
            warnings.warn(
                "Qwen向量召回与重排不可用，回退到词法召回: %s" % exc,
                RuntimeWarning,
            )
    return ToolRegistry(
        [
            SearchPoisTool(
                poi_repository,
                max_top_k=config.max_top_k,
                default_top_k=config.default_top_k,
            ),
            SearchNotesTool(
                note_repository,
                max_top_k=config.max_top_k,
                default_top_k=config.default_top_k,
            ),
            GetPoiNotesTool(note_repository),
            SearchNearbyTool(
                poi_repository,
                max_distance_m=config.max_nearby_distance_m,
                max_top_k=config.max_top_k,
                default_top_k=config.default_top_k,
            ),
        ]
    )
