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

from local_life_agent.agent import ReActAgent  # noqa: E402
from local_life_agent.config import AgentConfig, LLMConfig, PathConfig  # noqa: E402
from local_life_agent.dataset import load_tasks  # noqa: E402
from local_life_agent.llm import OpenAICompatibleClient  # noqa: E402
from local_life_agent.prompting import build_system_prompt  # noqa: E402
from local_life_agent.runner import run_batch  # noqa: E402
from local_life_agent.tools import build_default_registry  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="执行本地生活AI搜ReAct benchmark（不含judge）")
    parser.add_argument(
        "--input",
        type=Path,
        default=V5_DIR / "query" / "subset_1200_agent_input.json",
    )
    parser.add_argument("--output", type=Path, help="结果JSONL；默认写入outputs")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, help="本次选择的最大query数")
    parser.add_argument("--workers", type=int, default=1, help="并发数")
    parser.add_argument(
        "--max-tool-rounds", type=int, default=5, help="单query最大工具检索轮次（上限5）"
    )
    parser.add_argument(
        "--max-tools-per-round", type=int, default=2, help="每轮最大工具数（上限2）"
    )
    parser.add_argument(
        "--max-tool-calls", type=int, default=10, help="单query总工具调用上限（上限10）"
    )
    parser.add_argument(
        "--prompt",
        type=Path,
        help="系统提示词文件；默认用 PathConfig 内置的 prompts/react_system_prompt.md",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="覆盖指定输出并从头执行；默认按task_id断点续跑",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.offset < 0 or (args.limit is not None and args.limit < 1):
        raise SystemExit("--offset不能为负，--limit必须大于0")
    paths = (
        PathConfig(prompt_file=args.prompt) if args.prompt else PathConfig()
    )
    config = AgentConfig(
        max_tool_rounds=args.max_tool_rounds,
        max_tools_per_round=args.max_tools_per_round,
        max_tool_calls=args.max_tool_calls,
    )
    llm_config = LLMConfig.from_env()
    registry = build_default_registry(paths, config)
    system_prompt = build_system_prompt(registry, paths=paths, config=config)
    tasks = load_tasks(args.input)[args.offset :]
    if args.limit is not None:
        tasks = tasks[: args.limit]
    if args.output:
        output = args.output
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        output = paths.output_dir / ("react_run_%s.jsonl" % stamp)

    def agent_factory() -> ReActAgent:
        return ReActAgent(
            llm=OpenAICompatibleClient(llm_config),
            tools=registry,
            config=config,
            system_prompt=system_prompt,
            paths=paths,
        )

    print(
        json.dumps(
            {
                "input": str(args.input.resolve()),
                "selected_tasks": len(tasks),
                "output": str(output.resolve()),
                "model": llm_config.model,
                "api_style": llm_config.api_style,
                "workers": args.workers,
                "max_tool_rounds": args.max_tool_rounds,
                "max_tools_per_round": args.max_tools_per_round,
                "max_tool_calls": args.max_tool_calls,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    summary = run_batch(
        tasks,
        agent_factory=agent_factory,
        output_path=output,
        workers=args.workers,
        resume=not args.fresh,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
