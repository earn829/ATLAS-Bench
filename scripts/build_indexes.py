#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


V5_DIR = Path(__file__).resolve().parents[1]
if str(V5_DIR) not in sys.path:
    sys.path.insert(0, str(V5_DIR))

from local_life_agent.index_builder import build_indexes  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="拆分并构建POI/笔记两个独立RAG索引")
    parser.add_argument(
        "--source",
        type=Path,
        nargs="+",
        default=[V5_DIR.parent / "data" / "chaoyang_poi_features.json"],
        help="一个或多个源POI JSON数组，按给定顺序流式合并",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=V5_DIR / "data" / "indexes",
        help="索引输出目录",
    )
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有索引")
    parser.add_argument("--limit", type=int, help="仅构建前N个有效POI（开发调试）")
    parser.add_argument("--progress-every", type=int, default=10000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit 必须大于0")
    manifest = build_indexes(
        args.source,
        args.output_dir,
        overwrite=args.overwrite,
        limit=args.limit,
        progress_every=args.progress_every,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
