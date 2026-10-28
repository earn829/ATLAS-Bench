#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_INPUT = ROOT / "query" / "subset_1200_agent_input.json"
PYTHON = os.environ.get("PYTHON") or sys.executable
if "/" in PYTHON:
    PYTHON = str(Path(PYTHON).resolve())
# Read at call time so a shell that exports these after import still works.
HOSTED_BASE = os.environ.get("ATLAS_HOSTED_API_BASE", "")

PROFILES: Dict[str, Dict[str, Any]] = {
    "qwen35-local-no-think": {
        "slug": "qwen3.5-4B-local-nothinking-topk20-reclimit10",
        "model": "Qwen3.5-4B", "provider": "local", "thinking": False,
        "temperature": 0.1, "output": 8192, "workers": 16, "retry_workers": 4,
    },
    "qwen35-local-think": {
        "slug": "qwen3.5-4B-local-thinking-rp115-topk20-reclimit10",
        "model": "Qwen3.5-4B", "provider": "local", "thinking": True,
        "temperature": 0.1, "output": 8192, "retry_output": 16384,
        "repetition_penalty": 1.15, "workers": 16, "retry_workers": 4,
    },
    "qwen36-a3b-no-think": {
        "slug": "qwen3.6-35B-A3B-nothinking-topk20-reclimit10",
        "model": "Qwen3.6-35B-A3B", "provider": "local", "thinking": False,
        "temperature": 0.1, "output": 8192, "workers": 8, "retry_workers": 2,
    },
    "qwen36-a3b-think": {
        "slug": "qwen3.6-35B-A3B-thinking-topk20-reclimit10",
        "model": "Qwen3.6-35B-A3B", "provider": "local", "thinking": True,
        "temperature": 0.1, "output": 16384, "workers": 8, "retry_workers": 2,
    },
    "qwen38-local-no-think": {
        "slug": "qwen3.8-27B-local-nothinking-topk20-reclimit10",
        "model": "Qwen3.8-27B", "provider": "local", "thinking": False,
        "temperature": 0.1, "output": 8192, "workers": 16, "retry_workers": 4,
    },
    "qwen38-local-think": {
        "slug": "qwen3.8-27B-local-thinking-rp105-topk20-reclimit10",
        "model": "Qwen3.8-27B", "provider": "local", "thinking": True,
        "temperature": 0.1, "output": 8192, "retry_output": 16384,
        "repetition_penalty": 1.05, "workers": 12, "retry_workers": 4,
    },
    "qwen38-max": {
        "slug": "qwen3.8-max-hosted-default-thinking-topk20-reclimit10",
        "model": "qwen3.8-max", "provider": "hosted", "temperature": 0.1,
        "output": 8192, "workers": 12, "retry_workers": 3,
    },
    "deepseek-v41-flash": {
        "slug": "deepseek-v4-1-flash-hosted-topk20-reclimit10",
        "model": "deepseek-v4.1-flash", "provider": "hosted",
        "temperature": 0.1, "output": 8192, "workers": 6, "retry_workers": 3,
    },
    "gemini38-flash": {
        "slug": "gemini-3-8-flash-hosted-default-thinking-topk20-reclimit10",
        "model": "gemini-3.8-flash", "provider": "hosted", "temperature": 0.1,
        "output": 8192, "workers": 6, "retry_workers": 3,
    },
    "gpt56-luna": {
        "slug": "azure-gpt-5-6-luna-hosted-default-thinking-topk20-reclimit10",
        "model": "azure/gpt-5.6-luna", "provider": "hosted", "omit_temperature": True,
        "output": 8192, "workers": 12, "retry_workers": 3,
    },
    "gpt6-luna-xhigh": {
        "slug": "gpt-6-luna-hosted-xhigh-topk20-reclimit10",
        "model": "gpt-6-luna", "provider": "hosted", "omit_temperature": True,
        "reasoning_effort": "xhigh", "output": 8192, "workers": 12, "retry_workers": 3,
    },
    "glm53": {
        "slug": "glm-5-3-hosted-default-thinking-topk20-reclimit10",
        "model": "glm-5.3", "provider": "hosted", "temperature": 0.1,
        "output": 16384, "workers": 12, "retry_workers": 3,
    },
    "kimi-k3": {
        "slug": "kimi-k3-hosted-default-thinking-topk20-reclimit10",
        "model": "kimi-k3", "provider": "hosted", "omit_temperature": True,
        "output": 8192, "workers": 12, "retry_workers": 3,
    },
    "claude-sonnet5-bedrock": {
        "slug": "bedrock-claude-sonnet-5-hosted-default-thinking-cache-topk20-reclimit10",
        "model": "bedrock/claude-sonnet-5", "provider": "hosted",
        "omit_temperature": True, "cache_system_prompt": True,
        "output": 8192, "workers": 1, "retry_workers": 1, "api_retries": 8,
    },
}

CLEAR_KEYS = (
    "LOCAL_LIFE_API_URL", "API_URL", "LOCAL_LIFE_API_BASE",
    "LOCAL_LIFE_API_KEY", "LOCAL_LIFE_MODEL", "LOCAL_LIFE_API_STYLE",
    "LOCAL_LIFE_MAX_OUTPUT_TOKENS", "LOCAL_LIFE_TEMPERATURE",
    "LOCAL_LIFE_OMIT_TEMPERATURE", "LOCAL_LIFE_ENABLE_THINKING",
    "LOCAL_LIFE_THINKING_FIELD", "LOCAL_LIFE_REASONING_EFFORT",
    "LOCAL_LIFE_REPETITION_PENALTY", "LOCAL_LIFE_CACHE_SYSTEM_PROMPT",
    "LOCAL_LIFE_API_TIMEOUT", "LOCAL_LIFE_API_RETRIES",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a tested ATLAS-Bench model profile")
    parser.add_argument("profile", nargs="?", choices=sorted(PROFILES))
    parser.add_argument("stage", nargs="?", choices=("smoke", "full", "retry", "finalize", "status"))
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--reasoning-effort")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--list-profiles", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.input = args.input.resolve()
    return args


def paths(profile: Dict[str, Any], stage: str) -> tuple[Path, Path]:
    if stage == "smoke":
        directory = ROOT / "outputs" / ("_smoke_" + profile["slug"])
        return directory, directory / "smoke.jsonl"
    directory = ROOT / "outputs" / profile["slug"]
    return directory, directory / "subset_1200.jsonl"


def build_env(profile: Dict[str, Any], args: argparse.Namespace) -> Dict[str, str]:
    env = os.environ.copy()
    for key in CLEAR_KEYS:
        env.pop(key, None)
    env.update({
        "LOCAL_LIFE_API_STYLE": "chat_completions",
        "LOCAL_LIFE_MODEL": profile["model"],
        "LOCAL_LIFE_MAX_OUTPUT_TOKENS": str(
            args.max_output_tokens
            or (profile.get("retry_output") if args.stage == "retry" else None)
            or profile["output"]
        ),
        "LOCAL_LIFE_API_TIMEOUT": "900",
        "LOCAL_LIFE_API_RETRIES": str(profile.get("api_retries", 3)),
        "LOCAL_LIFE_VECTOR_DEVICE": "cuda",
        "LOCAL_LIFE_RERANKER_DEVICE": "cuda",
    })
    if profile["provider"] == "local":
        env["LOCAL_LIFE_API_BASE"] = os.environ.get(
            "ATLAS_LOCAL_API_BASE", "http://127.0.0.1:8001/v1")
        env["LOCAL_LIFE_API_KEY"] = os.environ.get("ATLAS_LOCAL_API_KEY", "local-vllm")
        env["LOCAL_LIFE_ENABLE_THINKING"] = str(profile["thinking"]).lower()
        env["LOCAL_LIFE_THINKING_FIELD"] = "chat_template_kwargs"
    else:
        key = os.environ.get("ATLAS_HOSTED_API_KEY") or os.environ.get("LOCAL_LIFE_API_KEY")
        if not key:
            raise SystemExit("ATLAS_HOSTED_API_KEY is required for this profile")
        base = (os.environ.get("ATLAS_HOSTED_API_URL")
                or os.environ.get("ATLAS_HOSTED_API_BASE") or HOSTED_BASE).rstrip("/")
        if not base:
            raise SystemExit("ATLAS_HOSTED_API_BASE or ATLAS_HOSTED_API_URL is required for hosted profiles")
        if base.endswith("/chat/completions"):
            base = base[: -len("/chat/completions")]
        env["LOCAL_LIFE_API_BASE"] = base
        env["LOCAL_LIFE_API_KEY"] = key
    if profile.get("omit_temperature"):
        env["LOCAL_LIFE_OMIT_TEMPERATURE"] = "true"
    else:
        env["LOCAL_LIFE_TEMPERATURE"] = str(profile["temperature"])
    effort = args.reasoning_effort or profile.get("reasoning_effort")
    if effort:
        env["LOCAL_LIFE_REASONING_EFFORT"] = effort
    if profile.get("repetition_penalty") is not None:
        env["LOCAL_LIFE_REPETITION_PENALTY"] = str(profile["repetition_penalty"])
    if profile.get("cache_system_prompt"):
        env["LOCAL_LIFE_CACHE_SYSTEM_PROMPT"] = "true"
    return env


def status(path: Path) -> int:
    if not path.exists():
        print(json.dumps({"path": str(path), "records": 0, "completed": 0, "failed": 0}))
        return 0
    best: Dict[str, Dict[str, Any]] = {}
    attempts: Counter[str] = Counter()
    parse_errors = 0
    for physical_line, line in enumerate(path.open(encoding="utf-8"), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            parse_errors += 1
            continue
        task_id = record.get("task_id")
        attempts[task_id] += 1
        previous = best.get(task_id)
        if previous is None or record.get("status") == "completed" or previous.get("status") != "completed":
            best[task_id] = record
    failures = Counter(
        (record.get("error") or {}).get("type", "Unknown")
        for record in best.values()
        if record.get("status") != "completed"
    )
    reasoning = completion = 0
    for record in best.values():
        if record.get("status") != "completed":
            continue
        for usage in (record.get("metrics") or {}).get("usage_events") or []:
            completion += usage.get("completion_tokens") or 0
            reasoning += (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
    result = {
        "path": str(path),
        "records": len(best),
        "completed": sum(r.get("status") == "completed" for r in best.values()),
        "failed": sum(r.get("status") != "completed" for r in best.values()),
        "failure_types": dict(failures),
        "tasks_with_multiple_attempts": sum(value > 1 for value in attempts.values()),
        "parse_errors": parse_errors,
        "reasoning_share": round(reasoning / completion, 6) if completion else 0.0,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    args = parse_args()
    if args.list_profiles:
        for name in sorted(PROFILES):
            print(name)
        return 0
    if not args.profile or not args.stage:
        raise SystemExit("profile and stage are required unless --list-profiles is used")
    profile = PROFILES[args.profile]
    directory, output = paths(profile, args.stage)
    if args.stage == "status":
        return status(output)
    if args.stage == "finalize":
        config = dict(profile)
        config["profile"] = args.profile
        command = [
            str(PYTHON), str(ROOT / "scripts" / "finalize_run.py"),
            "--input", str(output), "--tasks", str(args.input),
            "--output", str(directory / "subset_1200.final.jsonl"),
            "--config-json", json.dumps(config, ensure_ascii=False),
        ]
        return subprocess.run(command, cwd=ROOT).returncode
    if not args.input.exists():
        raise SystemExit(f"task input does not exist: {args.input}")
    directory.mkdir(parents=True, exist_ok=True)
    env = build_env(profile, args)
    workers = args.workers or (
        min(4, profile["workers"]) if args.stage == "smoke"
        else profile.get("retry_workers", 1) if args.stage == "retry"
        else profile["workers"]
    )
    command = [
        str(PYTHON), str(ROOT / "scripts" / "run_agent.py"),
        "--input", str(args.input), "--workers", str(workers), "--output", str(output),
    ]
    if args.stage == "smoke":
        command.extend(("--limit", "10"))
    if args.fresh:
        command.append("--fresh")
    if args.dry_run:
        safe_env = {key: value for key, value in env.items() if key.startswith("LOCAL_LIFE_") and key != "LOCAL_LIFE_API_KEY"}
        print(json.dumps({"env": safe_env, "command": command}, ensure_ascii=False, indent=2))
        return 0
    return subprocess.run(command, cwd=ROOT, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
