#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


V5_DIR = Path(__file__).resolve().parents[1]
if str(V5_DIR) not in sys.path:
    sys.path.insert(0, str(V5_DIR))

from local_life_agent.dense_retrieval import (  # noqa: E402
    DEFAULT_DIMENSION,
    DEFAULT_MAX_SEQ_LENGTH,
    DEFAULT_MODEL_NAME,
    DEFAULT_RERANKER_MAX_LENGTH,
    DEFAULT_RERANKER_MODEL_NAME,
)
from local_life_agent.vector_builder import build_vector_indexes  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="为v5构建POI和笔记的FAISS向量索引")
    parser.add_argument(
        "--index-dir", type=Path, default=V5_DIR / "data" / "indexes"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--backend", choices=["torch", "mlx"], default="torch"
    )
    parser.add_argument(
        "--mlx-model-path", type=Path, help="转换后的本地MLX模型目录"
    )
    parser.add_argument("--dimension", type=int, default=DEFAULT_DIMENSION)
    parser.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL_NAME)
    parser.add_argument(
        "--reranker-max-length", type=int, default=DEFAULT_RERANKER_MAX_LENGTH
    )
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda", "mps"], default="auto"
    )
    parser.add_argument("--poi-batch-size", type=int, default=16)
    parser.add_argument("--note-batch-size", type=int, default=8)
    parser.add_argument(
        "--workers", type=int, default=1, help="并行模型进程数；CPU建议2"
    )
    parser.add_argument(
        "--shards", type=int, default=64, help="每个库的可恢复分片数"
    )
    parser.add_argument("--progress-every", type=int, default=5000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = build_vector_indexes(
        args.index_dir,
        model_name=args.model,
        embedding_backend=args.backend,
        mlx_model_path=args.mlx_model_path,
        dimension=args.dimension,
        max_seq_length=args.max_seq_length,
        reranker_model_name=args.reranker_model,
        reranker_max_length=args.reranker_max_length,
        device=args.device,
        poi_batch_size=args.poi_batch_size,
        note_batch_size=args.note_batch_size,
        workers=args.workers,
        shard_count=args.shards,
        progress_every=args.progress_every,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
