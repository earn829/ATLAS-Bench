#!/usr/bin/env python3
"""Fetch the pinned Qwen3 embedding and reranker models into data/models.

The package ships the SQLite and FAISS indexes but not the 2.3 GB of weights.
Those indexes were built with the exact revisions pinned below: a different
commit changes query embeddings and silently degrades retrieval, because
vector_manifest.json records the model name but not the revision.

Honours HF_ENDPOINT, so a mirror can be used by exporting it before running.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / "data" / "models"
PINNED: Dict[str, str] = {
    "Qwen/Qwen3-Embedding-0.6B": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
    "Qwen/Qwen3-Reranker-0.6B": "e61197ed45024b0ed8a2d74b80b4d909f1255473",
}


def cache_root(name: str, cache_dir: Path) -> Path:
    return cache_dir / ("models--" + name.replace("/", "--"))


def local_revision(name: str, cache_dir: Path) -> Optional[str]:
    ref = cache_root(name, cache_dir) / "refs" / "main"
    if not ref.is_file():
        return None
    revision = ref.read_text(encoding="utf-8").strip()
    snapshot = cache_root(name, cache_dir) / "snapshots" / revision
    return revision if snapshot.is_dir() else None


def report(cache_dir: Path) -> bool:
    complete = True
    for name, pinned in PINNED.items():
        found = local_revision(name, cache_dir)
        if found == pinned:
            print(f"OK       {name}  {pinned[:12]}")
        elif found is None:
            complete = False
            print(f"MISSING  {name}  need {pinned[:12]}")
        else:
            complete = False
            print(f"MISMATCH {name}  local {found[:12]} != pinned {pinned[:12]}")
    return complete


def fetch(cache_dir: Path) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit(
            "缺少 huggingface_hub；先安装 requirements-vector.txt（sentence-transformers 会带上它）"
        )
    cache_dir.mkdir(parents=True, exist_ok=True)
    for name, revision in PINNED.items():
        if local_revision(name, cache_dir) == revision:
            print(f"SKIP     {name}  already at {revision[:12]}")
            continue
        print(f"FETCH    {name}  revision {revision[:12]} -> {cache_dir}")
        snapshot_download(repo_id=name, revision=revision, cache_dir=str(cache_dir))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE,
                        help=f"HF cache holding models--* directories (default: {DEFAULT_CACHE})")
    parser.add_argument("--check", action="store_true",
                        help="Only report whether the pinned revisions are present; exit 1 if not")
    args = parser.parse_args(argv)
    cache_dir = args.cache_dir.resolve()

    if args.check:
        return 0 if report(cache_dir) else 1

    fetch(cache_dir)
    if not report(cache_dir):
        print("\n模型仍不完整；检索会在运行时报错而不是静默降级。", file=sys.stderr)
        return 1
    print(f"\n模型就绪：{cache_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
