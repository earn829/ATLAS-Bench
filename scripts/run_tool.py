#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


V5_DIR = Path(__file__).resolve().parents[1]
if str(V5_DIR) not in sys.path:
    sys.path.insert(0, str(V5_DIR))

from local_life_agent.tools import build_default_registry  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="手工调用一个已注册工具")
    parser.add_argument(
        "tool",
        choices=["search_pois", "search_notes", "get_poi_notes", "search_nearby"],
    )
    parser.add_argument("arguments", help="工具arguments的JSON object")
    args = parser.parse_args()
    try:
        arguments = json.loads(args.arguments)
    except json.JSONDecodeError as exc:
        raise SystemExit("arguments不是合法JSON: %s" % exc)
    if not isinstance(arguments, dict):
        raise SystemExit("arguments顶层必须是JSON object")
    result = build_default_registry().execute(args.tool, arguments)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
