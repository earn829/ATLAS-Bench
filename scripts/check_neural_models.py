#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


V5_DIR = Path(__file__).resolve().parents[1]
if str(V5_DIR) not in sys.path:
    sys.path.insert(0, str(V5_DIR))

from local_life_agent.dense_retrieval import (  # noqa: E402
    DEFAULT_DIMENSION,
    DEFAULT_MAX_SEQ_LENGTH,
    DEFAULT_MODEL_NAME,
    DEFAULT_QUERY_PROMPT_NAME,
    DEFAULT_RERANKER_MAX_LENGTH,
    DEFAULT_RERANKER_MODEL_NAME,
    SharedTextEmbedder,
    embedding_cache_dir,
)
from local_life_agent.reranking import SharedQwen3Reranker  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="检查v5 Qwen向量与重排模型")
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda", "mps"], default="auto"
    )
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    embedder = SharedTextEmbedder(
        model_name=DEFAULT_MODEL_NAME,
        dimension=DEFAULT_DIMENSION,
        query_prompt_name=DEFAULT_QUERY_PROMPT_NAME,
        max_seq_length=DEFAULT_MAX_SEQ_LENGTH,
        cache_dir=embedding_cache_dir(),
        device=args.device,
        local_files_only=args.local_files_only,
    )
    query_vector = embedder.encode_query("望京附近安静的咖啡店")
    document_vectors = embedder.encode_documents(
        ["名称：安静角落咖啡；商圈：望京", "名称：望京地铁站；类别：地铁站"],
        batch_size=2,
    )
    embedding_elapsed = time.perf_counter() - started

    reranker_started = time.perf_counter()
    reranker = SharedQwen3Reranker(
        model_name=DEFAULT_RERANKER_MODEL_NAME,
        max_length=DEFAULT_RERANKER_MAX_LENGTH,
        batch_size=4,
        cache_dir=embedding_cache_dir(),
        device=args.device,
        local_files_only=args.local_files_only,
    )
    rerank_scores = reranker.score(
        "望京附近安静的咖啡店",
        ["名称：安静角落咖啡；商圈：望京", "名称：望京地铁站；类别：地铁站"],
    )
    reranker_elapsed = time.perf_counter() - reranker_started

    print(
        json.dumps(
            {
                "embedding_model": DEFAULT_MODEL_NAME,
                "embedding_device": embedder.device,
                "embedding_shape": list(query_vector.shape),
                "embedding_similarities": (
                    query_vector @ document_vectors.T
                )[0].tolist(),
                "embedding_elapsed_seconds": round(embedding_elapsed, 3),
                "reranker_model": DEFAULT_RERANKER_MODEL_NAME,
                "reranker_device": reranker.device,
                "rerank_scores": rerank_scores,
                "reranker_elapsed_seconds": round(reranker_elapsed, 3),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
