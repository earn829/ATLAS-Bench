from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


VECTOR_SCHEMA_VERSION = 2
DEFAULT_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_DIMENSION = 1024
DEFAULT_MAX_SEQ_LENGTH = 512
DEFAULT_QUERY_PROMPT_NAME = "query"
DEFAULT_RERANKER_MODEL_NAME = "Qwen/Qwen3-Reranker-0.6B"
DEFAULT_RERANKER_MAX_LENGTH = 512
DEFAULT_MLX_MODEL_DIR_NAME = "Qwen3-Embedding-0.6B-mlx-fp16"


class VectorIndexError(RuntimeError):
    pass


def resolve_model_source(
    model_name: str, cache_dir: Path, *, local_files_only: bool
) -> str:
    direct = Path(model_name).expanduser()
    if direct.is_dir():
        return str(direct.resolve())
    model_root = Path(cache_dir) / ("models--" + model_name.replace("/", "--"))
    ref_path = model_root / "refs" / "main"
    if ref_path.is_file():
        revision = ref_path.read_text(encoding="utf-8").strip()
        snapshot = model_root / "snapshots" / revision
        if revision and snapshot.is_dir():
            return str(snapshot.resolve())
    if local_files_only:
        raise VectorIndexError("本地模型snapshot不存在: %s" % model_name)
    return model_name


def load_vector_manifest(
    index_dir: Path,
    *,
    expected_source_build_id: str,
    expected_poi_count: int,
    expected_note_count: int,
) -> Dict[str, Any]:
    path = Path(index_dir) / "vector_manifest.json"
    if not path.is_file():
        raise FileNotFoundError("向量索引manifest不存在: %s" % path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VectorIndexError("向量索引manifest无法读取: %s" % path) from exc
    checks = (
        ("vector_schema_version", VECTOR_SCHEMA_VERSION),
        ("source_build_id", expected_source_build_id),
        ("poi_count", expected_poi_count),
        ("note_count", expected_note_count),
    )
    for key, expected in checks:
        if value.get(key) != expected:
            raise VectorIndexError(
                "向量索引manifest的%s=%r，期望=%r；请重建向量索引"
                % (key, value.get(key), expected)
            )
    for key in (
        "model_name",
        "embedding_backend",
        "dimension",
        "query_prompt_name",
        "max_seq_length",
        "reranker",
    ):
        if key not in value:
            raise VectorIndexError("向量索引manifest缺少字段: %s" % key)
    if value["embedding_backend"] not in {"torch", "mlx"}:
        raise VectorIndexError("向量索引manifest的embedding_backend不受支持")
    reranker = value["reranker"]
    if not isinstance(reranker, dict):
        raise VectorIndexError("向量索引manifest的reranker必须是object")
    for key in ("model_name", "max_length", "score_activation"):
        if key not in reranker:
            raise VectorIndexError("向量索引manifest的reranker缺少字段: %s" % key)
    return value


class SharedTextEmbedder:
    """Lazily loaded sentence-transformers encoder shared by both indexes."""

    def __init__(
        self,
        *,
        model_name: str,
        dimension: int,
        query_prompt_name: str,
        max_seq_length: int,
        cache_dir: Path,
        device: str = "auto",
        local_files_only: bool = False,
    ) -> None:
        self.model_name = model_name
        self.dimension = int(dimension)
        self.query_prompt_name = query_prompt_name
        self.max_seq_length = int(max_seq_length)
        self.cache_dir = Path(cache_dir)
        self.requested_device = device
        self.local_files_only = local_files_only
        self._model = None
        self._device = ""
        self._lock = threading.Lock()

    @property
    def device(self) -> str:
        self._load()
        return self._device

    def _resolve_device(self) -> str:
        if self.requested_device != "auto":
            return self.requested_device
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    def _load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise VectorIndexError(
                    "缺少sentence-transformers；请使用.venv或安装requirements-vector.txt"
                ) from exc
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._device = self._resolve_device()
            model_source = resolve_model_source(
                self.model_name,
                self.cache_dir,
                local_files_only=self.local_files_only,
            )
            model = SentenceTransformer(
                model_source,
                device=self._device,
                cache_folder=str(self.cache_dir),
                local_files_only=self.local_files_only,
            )
            model.max_seq_length = self.max_seq_length
            if hasattr(model, "tokenizer"):
                # Qwen3's left-padded batches can produce NaNs on PyTorch MPS.
                # The sentence-transformers Lasttoken pooling module honors the
                # attention mask, so right padding remains semantically correct.
                model.tokenizer.padding_side = (
                    "right" if self._device == "mps" else "left"
                )
            output_dimension = int(model.get_sentence_embedding_dimension())
            if self.dimension > output_dimension:
                raise VectorIndexError(
                    "配置向量维度%d超过模型输出维度%d"
                    % (self.dimension, output_dimension)
                )
            self._model = model

    def _encode(
        self,
        texts: Sequence[str],
        *,
        is_query: bool,
        batch_size: int,
    ) -> np.ndarray:
        self._load()
        inputs = [str(text) for text in texts]
        encode_kwargs: Dict[str, Any] = {}
        if is_query and self.query_prompt_name:
            prompts = getattr(self._model, "prompts", {})
            if self.query_prompt_name not in prompts:
                raise VectorIndexError(
                    "embedding模型缺少query prompt: %s" % self.query_prompt_name
                )
            encode_kwargs["prompt_name"] = self.query_prompt_name
        with self._lock:
            vectors = self._model.encode(
                inputs,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False,
                **encode_kwargs,
            )
        array = np.asarray(vectors, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        array = np.ascontiguousarray(array[:, : self.dimension], dtype=np.float32)
        if not np.isfinite(array).all():
            raise VectorIndexError(
                "embedding模型生成NaN或Inf；请检查设备精度和padding设置"
            )
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        np.divide(array, np.maximum(norms, 1e-12), out=array)
        return array

    def encode_query(self, query: str) -> np.ndarray:
        return self._encode([query], is_query=True, batch_size=1)

    def encode_documents(
        self, texts: Sequence[str], *, batch_size: int
    ) -> np.ndarray:
        return self._encode(texts, is_query=False, batch_size=batch_size)


class SharedMlxTextEmbedder:
    """Qwen3 embedding inference through Apple's native MLX runtime."""

    def __init__(
        self,
        *,
        model_name: str,
        model_path: Path,
        dimension: int,
        query_prompt_name: str,
        max_seq_length: int,
    ) -> None:
        self.model_name = model_name
        self.model_path = Path(model_path)
        self.dimension = int(dimension)
        self.query_prompt_name = query_prompt_name
        self.max_seq_length = int(max_seq_length)
        self._model = None
        self._tokenizer = None
        self._query_prompt = ""
        self._lock = threading.Lock()

    @property
    def device(self) -> str:
        self._load()
        return "mlx"

    def _load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            if not self.model_path.is_dir():
                raise FileNotFoundError("MLX embedding模型不存在: %s" % self.model_path)
            try:
                from mlx_embeddings.utils import load
            except ImportError as exc:
                raise VectorIndexError(
                    "缺少mlx-embeddings；请使用Python>=3.10安装requirements-mlx.txt"
                ) from exc
            prompt_path = self.model_path / "config_sentence_transformers.json"
            try:
                prompt_config = json.loads(prompt_path.read_text(encoding="utf-8"))
                prompts = prompt_config.get("prompts", {})
                self._query_prompt = str(prompts[self.query_prompt_name])
            except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
                raise VectorIndexError(
                    "MLX模型缺少query prompt配置: %s" % prompt_path
                ) from exc
            self._model, self._tokenizer = load(str(self.model_path))

    def _encode(
        self,
        texts: Sequence[str],
        *,
        is_query: bool,
        batch_size: int,
    ) -> np.ndarray:
        self._load()
        values = [str(text) for text in texts]
        if is_query:
            values = [self._query_prompt + value for value in values]
        batches: List[np.ndarray] = []
        try:
            import mlx.core as mx
        except ImportError as exc:
            raise VectorIndexError("缺少Apple MLX运行时") from exc
        with self._lock:
            for offset in range(0, len(values), batch_size):
                inputs = self._tokenizer(
                    values[offset : offset + batch_size],
                    return_tensors="np",
                    padding=True,
                    truncation=True,
                    max_length=self.max_seq_length,
                )
                output = self._model(
                    mx.array(inputs["input_ids"]),
                    attention_mask=mx.array(inputs["attention_mask"]),
                ).text_embeds
                mx.eval(output)
                batches.append(np.asarray(output, dtype=np.float32))
        if not batches:
            return np.empty((0, self.dimension), dtype=np.float32)
        array = np.ascontiguousarray(
            np.concatenate(batches, axis=0)[:, : self.dimension], dtype=np.float32
        )
        if not np.isfinite(array).all():
            raise VectorIndexError("MLX embedding模型生成NaN或Inf")
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        np.divide(array, np.maximum(norms, 1e-12), out=array)
        return array

    def encode_query(self, query: str) -> np.ndarray:
        return self._encode([query], is_query=True, batch_size=1)

    def encode_documents(
        self, texts: Sequence[str], *, batch_size: int
    ) -> np.ndarray:
        return self._encode(texts, is_query=False, batch_size=batch_size)


class FaissVectorIndex:
    """Read-only normalized inner-product index returning SQLite row IDs."""

    def __init__(
        self,
        path: Path,
        *,
        embedder: SharedTextEmbedder,
        expected_count: int,
        label: str,
    ) -> None:
        try:
            import faiss
        except ImportError as exc:
            raise VectorIndexError(
                "缺少faiss-cpu；请使用.venv或安装requirements-vector.txt"
            ) from exc
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError("%s不存在: %s" % (label, path))
        self.path = path
        self.embedder = embedder
        self.label = label
        self.index = faiss.read_index(str(path))
        if self.index.d != embedder.dimension:
            raise VectorIndexError(
                "%s维度=%d，模型配置维度=%d"
                % (label, self.index.d, embedder.dimension)
            )
        if self.index.ntotal != expected_count:
            raise VectorIndexError(
                "%s条数=%d，期望=%d" % (label, self.index.ntotal, expected_count)
            )

    def search_ids(
        self,
        query: str,
        top_k: int,
        *,
        eligible_ids: Optional[Sequence[int]] = None,
    ) -> List[Tuple[int, float]]:
        if top_k < 1:
            return []
        params = None
        available = int(self.index.ntotal)
        eligible_array = None
        if eligible_ids is not None:
            eligible_array = np.ascontiguousarray(
                list(dict.fromkeys(int(value) for value in eligible_ids)),
                dtype=np.int64,
            )
            available = int(eligible_array.size)
            if available == 0:
                return []
            try:
                import faiss
            except ImportError as exc:
                raise VectorIndexError("缺少faiss-cpu") from exc
            params = faiss.SearchParameters()
            params.sel = faiss.IDSelectorBatch(eligible_array)
        query_vector = self.embedder.encode_query(query)
        scores, identifiers = self.index.search(
            query_vector,
            min(int(top_k), available),
            params=params,
        )
        return [
            (int(identifier), float(score))
            for identifier, score in zip(identifiers[0], scores[0])
            if int(identifier) >= 0
        ]


def embedding_cache_dir() -> Path:
    configured = os.environ.get("LOCAL_LIFE_VECTOR_CACHE", "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path(__file__).resolve().parents[1] / "data" / "models"


def mlx_embedding_model_dir(name: str = DEFAULT_MLX_MODEL_DIR_NAME) -> Path:
    configured = os.environ.get("LOCAL_LIFE_MLX_MODEL_PATH", "").strip()
    if configured:
        return Path(configured).expanduser()
    return embedding_cache_dir() / name
