"""Offline contract tests for a relocated ATLAS-Bench checkout.

Only temporary copies and synthetic two-task fixtures are executed. Native judge
calls are mocked; shell drivers use a recording interpreter, never a scoring API.
The legacy maps and metric expectations below are fixed contract snapshots, not
values discovered from the implementation under test.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock
import urllib.request


PACKAGE = Path(__file__).resolve().parents[1]
PROFILE_REL = Path(".qoder/skills/atlas-benchmark-eval/scripts/run_profile.py")
AGENT_TASKS = "subset_1200_agent_input.json"
RAW_TASKS = "subset_1200.json"
LEGACY_RUNS = {
    "qwen35_4b_nothink": "qwen3.5-4B-local-nothinking-topk20-reclimit10",
    "qwen35_4b_think": "qwen3.5-4B-local-thinking-rp115-topk20-reclimit10",
    "qwen36_a3b_nothink": "qwen3.6-35B-A3B-nothinking-topk20-reclimit10",
    "qwen36_a3b_think": "qwen3.6-35B-A3B-thinking-topk20-reclimit10",
    "qwen38_27b_nothink": "qwen3.8-27B-local-nothinking-topk20-reclimit10",
    "qwen38_27b_think": "qwen3.8-27B-local-thinking-rp105-topk20-reclimit10",
    "qwen38_max": "qwen3.8-max-hosted-default-thinking-topk20-reclimit10",
    "deepseek_v41_flash": "deepseek-v4-1-flash-hosted-topk20-reclimit10",
    "claude5_sonnet": "bedrock-claude-sonnet-5-hosted-default-thinking-cache-topk20-reclimit10",
    "gpt56_luna": "azure-gpt-5-6-luna-hosted-default-thinking-topk20-reclimit10",
    "gpt6_luna": "gpt-6-luna-hosted-xhigh-topk20-reclimit10",
    "glm53": "glm-5-3-hosted-default-thinking-topk20-reclimit10",
    "kimi_k3": "kimi-k3-hosted-default-thinking-topk20-reclimit10",
    "gemini38_flash": "gemini-3-8-flash-hosted-default-thinking-topk20-reclimit10",
}
# wrapper suffix -> (native script path, legacy output folder, workers, score field)
DIMENSIONS = {
    "factual_accuracy": ("judge/factual_accuracy_judge.py", "factual_accuracy_deepseek_pro", 20, "factual_correct"),
    "intent_understanding": ("judge/intent_understanding_judge.py", "intent_understanding_deepseek_pro", 40, "intent_score"),
    "tool_selection": ("judge/tool_selection_judge.py", "tool_selection_deepseek_pro", 40, "tool_score"),
    "evidence_sufficiency": ("judge/evidence_sufficiency_judge.py", "evidence_sufficiency_deepseek_pro", 40, "evidence_score"),
    "tool_flexibility": ("judge/tool_flexibility_judge.py", "tool_flexibility_deepseek_pro", 20, "flexibility_score"),
}
TASKFUL = {"intent_understanding", "tool_selection", "evidence_sufficiency"}

# These are the original profile parameters, including deliberately different
# retry budgets/concurrency and provider-specific generation settings.
PROFILES = {
    "qwen35-local-no-think": dict(slug=LEGACY_RUNS["qwen35_4b_nothink"], model="Qwen3.5-4B", provider="local", thinking=False, temperature=0.1, output=8192, workers=16, retry_workers=4),
    "qwen35-local-think": dict(slug=LEGACY_RUNS["qwen35_4b_think"], model="Qwen3.5-4B", provider="local", thinking=True, temperature=0.1, output=8192, retry_output=16384, repetition_penalty=1.15, workers=16, retry_workers=4),
    "qwen36-a3b-no-think": dict(slug=LEGACY_RUNS["qwen36_a3b_nothink"], model="Qwen3.6-35B-A3B", provider="local", thinking=False, temperature=0.1, output=8192, workers=8, retry_workers=2),
    "qwen36-a3b-think": dict(slug=LEGACY_RUNS["qwen36_a3b_think"], model="Qwen3.6-35B-A3B", provider="local", thinking=True, temperature=0.1, output=16384, workers=8, retry_workers=2),
    "qwen38-local-no-think": dict(slug=LEGACY_RUNS["qwen38_27b_nothink"], model="Qwen3.8-27B", provider="local", thinking=False, temperature=0.1, output=8192, workers=16, retry_workers=4),
    "qwen38-local-think": dict(slug=LEGACY_RUNS["qwen38_27b_think"], model="Qwen3.8-27B", provider="local", thinking=True, temperature=0.1, output=8192, retry_output=16384, repetition_penalty=1.05, workers=12, retry_workers=4),
    "qwen38-max": dict(slug=LEGACY_RUNS["qwen38_max"], model="qwen3.8-max", provider="hosted", temperature=0.1, output=8192, workers=12, retry_workers=3),
    "deepseek-v41-flash": dict(slug=LEGACY_RUNS["deepseek_v41_flash"], model="deepseek-v4.1-flash", provider="hosted", temperature=0.1, output=8192, workers=6, retry_workers=3),
    "gemini38-flash": dict(slug=LEGACY_RUNS["gemini38_flash"], model="gemini-3.8-flash", provider="hosted", temperature=0.1, output=8192, workers=6, retry_workers=3),
    "gpt56-luna": dict(slug=LEGACY_RUNS["gpt56_luna"], model="azure/gpt-5.6-luna", provider="hosted", omit_temperature=True, output=8192, workers=12, retry_workers=3),
    "gpt6-luna-xhigh": dict(slug=LEGACY_RUNS["gpt6_luna"], model="gpt-6-luna", provider="hosted", omit_temperature=True, reasoning_effort="xhigh", output=8192, workers=12, retry_workers=3),
    "glm53": dict(slug=LEGACY_RUNS["glm53"], model="glm-5.3", provider="hosted", temperature=0.1, output=16384, workers=12, retry_workers=3),
    "kimi-k3": dict(slug=LEGACY_RUNS["kimi_k3"], model="kimi-k3", provider="hosted", omit_temperature=True, output=8192, workers=12, retry_workers=3),
    "claude-sonnet5-bedrock": dict(slug=LEGACY_RUNS["claude5_sonnet"], model="bedrock/claude-sonnet-5", provider="hosted", omit_temperature=True, cache_system_prompt=True, output=8192, workers=1, retry_workers=1, api_retries=8),
}


def json_stream(text):
    """Decode pretty-printed consecutive JSON values, without forgiving log noise."""
    decoder = json.JSONDecoder()
    values = []
    rest = text.strip()
    while rest:
        value, end = decoder.raw_decode(rest)
        values.append(value)
        rest = rest[end:].strip()
    return values


@contextlib.contextmanager
def working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class RelocatedCase(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="atlas portability ")
        self.addCleanup(temporary.cleanup)
        self.temp = Path(temporary.name)
        self.root = self.temp / "relocated checkout with spaces"
        self.caller = self.temp / "unrelated caller with spaces"
        self.root.mkdir()
        self.caller.mkdir()
        # No original outputs, indices, trajectory, .env, .venv, or paper files.
        for directory, patterns in (("scripts", ("*.py", "*.sh")), ("judge", ("*.py",)),
                                    ("local_life_agent", ("**/*.py",))):
            for pattern in patterns:
                for source in (PACKAGE / directory).glob(pattern):
                    destination = self.root / source.relative_to(PACKAGE)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
        for relative in (PROFILE_REL, Path("prompts/react_system_prompt.md")):
            destination = self.root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(PACKAGE / relative, destination)
        self.env = {
            "PATH": os.defpath,
            "HOME": str(self.temp),
            "LANG": "C.UTF-8",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        }
        self.make_fixtures()
        self.log = self.temp / "recorded interpreter calls.jsonl"
        self.fake_python = self.caller / "tools with spaces" / "python with spaces"
        self.make_fake_python(self.fake_python)
        self.env.update(PYTHON=str(self.fake_python), PORTABILITY_ARGV_LOG=str(self.log))
        # This guard also protects in-process imports and mocks against regressions.
        for target in ("urllib.request.urlopen", "socket.create_connection", "socket.socket.connect"):
            patch = mock.patch(target, side_effect=AssertionError("network is forbidden in portability tests"))
            patch.start()
            self.addCleanup(patch.stop)

    def write_json(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        return path

    def write_jsonl(self, path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        return path

    def make_fixtures(self):
        context = {"time": "2026-01-01 12:00:00", "loc": {"lng": 116.4, "lat": 39.9}}
        constraints = [
            {"cid": "c1", "text": "人均不超过50元", "h": "刚性", "s": "明确", "evidence_type": "poi", "expected_predicate": {"average_cost_lte": 50}, "tool": "poi.search", "private_annotation": "DO_NOT_EXPOSE"},
            {"cid": "c2", "text": "适合安静阅读", "h": "柔性", "s": "偏好", "evidence_type": "note", "expected_predicate": {"quiet": True}, "tool": "note.search"},
        ]
        self.tasks = [
            {"id": "task-full", "query": "找适合阅读的咖啡店", "context": context, "constraints": constraints, "structure": {"private": "DO_NOT_EXPOSE"}, "answer": {"satisfiable": True, "reference": "DO_NOT_EXPOSE"}, "target_poi": "DO_NOT_EXPOSE"},
            {"id": "task-empty", "query": "找不存在的店", "context": context, "constraints": [], "structure": {}, "answer": {"satisfiable": False}},
        ]
        self.raw_tasks = self.write_json(self.root / "query" / RAW_TASKS, [{"record": task} for task in self.tasks])
        self.agent_tasks = self.write_json(self.root / "query" / AGENT_TASKS, {"records": [{k: task[k] for k in ("id", "query", "context")} for task in self.tasks]})
        self.observed = [
            {"poiid": "p1", "name": "第一咖啡店", "evidence_id": "poi:p1", "rating": 4.5, "average_cost": 40},
            {"poiid": "p2", "name": "第二咖啡店", "evidence_id": "poi:p2", "rating": 4.0, "average_cost": 45},
            {"poiid": "p3", "name": "第三咖啡店", "evidence_id": "poi:p3", "rating": 3.5, "average_cost": 60},
            {"poiid": "not-recommended", "name": "仅观察的店", "evidence_id": "poi:extra", "average_cost": 80},
            {"poiid": "p1", "evidence_id": "note:p1", "content": "阅读很安静", "open_tag": "安静"},
        ]
        recommendations = [
            {"poiid": "p1", "name": "第一咖啡店", "reason": "人均40元，笔记提到安静", "evidence": ["poi:p1", "note:p1"], "constraint_gaps": []},
            {"poiid": "p2", "name": "第二咖啡店", "reason": "人均45元", "evidence": ["poi:p2"], "constraint_gaps": ["安静程度未知"]},
            {"poiid": "p3", "name": "第三咖啡店", "reason": "人均60元", "evidence": ["poi:p3"], "constraint_gaps": ["超过预算"]},
        ]
        self.records = [
            {"task_id": "task-full", "status": "completed", "input": {"query": self.tasks[0]["query"], **context}, "model": "DO_NOT_EXPOSE", "answer": {"summary": "推荐三家店", "recommendations": recommendations, "no_exact_match": False, "notes": ["另有一家人均80元"]}, "trajectory": [{"event": "tool_calls", "round": 1, "calls": [{"name": "poi.search", "arguments": {"query": "咖啡"}, "observation": {"data": {"results": [{**item, "private": "DO_NOT_EXPOSE"} for item in self.observed[:4]]}}}, {"name": "note.search", "arguments": {"poiid": "p1"}, "observation": {"data": {"notes": [self.observed[4]]}}}]}], "metrics": {"tool_calls": 2, "elapsed_seconds": 1.25}},
            {"task_id": "task-empty", "status": "completed", "input": {"query": self.tasks[1]["query"], **context}, "answer": {"summary": "没有匹配", "recommendations": [], "no_exact_match": True, "notes": []}, "trajectory": [], "metrics": {}},
        ]
        self.run = self.write_jsonl(self.caller / "input data" / "fixture run.jsonl", self.records)

    def expected_compact(self, index=0):
        result = []
        by_evidence = {item["evidence_id"]: item for item in self.observed}
        for rank, rec in enumerate(self.records[index]["answer"]["recommendations"], 1):
            result.append({
                "rank": rank, "poiid": rec["poiid"], "name": rec["name"], "reason": rec["reason"],
                "constraint_gaps": rec["constraint_gaps"], "evidence_ids": rec["evidence"],
                "cited_evidence": [by_evidence[eid] for eid in rec["evidence"]],
                "observed_facts": [item for item in self.observed if item["poiid"] == rec["poiid"]],
            })
        return result

    def expected_qoder(self, index):
        task = self.tasks[index]
        fields = ("cid", "text", "h", "s", "evidence_type", "expected_predicate")
        return {"task_id": task["id"], "query": task["query"], "context": task["context"],
                "constraints": [{key: constraint.get(key) for key in fields} for constraint in task["constraints"]],
                "n_constraints": len(task["constraints"]), "satisfiable": task["answer"]["satisfiable"],
                "recommendations": self.expected_compact(index)}

    def expected_factual(self, index):
        record = self.records[index]
        answer = record["answer"]
        return {"task_id": record["task_id"], "query": record["input"]["query"],
                "context": {key: value for key, value in record["input"].items() if key != "query"},
                "answer": {"summary": answer["summary"], "recommendations": self.expected_compact(index),
                           "no_exact_match": answer["no_exact_match"], "notes": answer["notes"]},
                "all_observed_evidence": self.observed if index == 0 else []}

    def make_fake_python(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "#!" + sys.executable + "\n"
            "import json, os, subprocess, sys\n"
            "if len(sys.argv) > 1 and sys.argv[1] == '-c':\n"
            "    raise SystemExit(subprocess.run([" + repr(sys.executable) + ", *sys.argv[1:]]).returncode)\n"
            "with open(os.environ['PORTABILITY_ARGV_LOG'], 'a', encoding='utf-8') as handle:\n"
            "    handle.write(json.dumps({'argv': sys.argv[1:], 'cwd': os.getcwd()}) + '\\n')\n",
            encoding="utf-8",
        )
        path.chmod(0o755)

    def calls(self):
        return [json.loads(line) for line in self.log.read_text(encoding="utf-8").splitlines()] if self.log.exists() else []

    def clear_calls(self):
        if self.log.exists():
            self.log.unlink()

    def command(self, relative, *args, env=None):
        executable = shutil.which("bash", path=os.defpath) if str(relative).endswith(".sh") else sys.executable
        self.assertIsNotNone(executable, "bash is required to test the shell drivers")
        return subprocess.run([executable, str(self.root / relative), *map(str, args)], cwd=self.caller,
                              env=self.env if env is None else env, text=True, capture_output=True, timeout=30)

    def successful(self, result):
        self.assertEqual(result.returncode, 0, (result.stdout + "\n" + result.stderr)[-5000:])

    def option(self, call, flag):
        argv = call["argv"] if isinstance(call, dict) else list(map(str, call))
        self.assertIn(flag, argv)
        self.assertEqual(argv.count(flag), 1, argv)
        return argv[argv.index(flag) + 1]

    def argument_path(self, call, flag=None):
        value = self.option(call, flag) if flag else call["argv"][0]
        path = Path(value)
        return (path if path.is_absolute() else Path(call["cwd"]) / path).resolve()

    def judge_artifacts(self):
        """Non-code entries under the relocated judge/ directory (i.e. scoring results)."""
        judge = self.root / "judge"
        if not judge.exists():
            return []
        return sorted(str(path.relative_to(judge)) for path in judge.rglob("*") if path.suffix != ".py")

    def tree(self, directory):
        return {str(path.relative_to(directory)): path.read_bytes() for path in directory.rglob("*") if path.is_file()} if directory.exists() else {}

    @contextlib.contextmanager
    def module(self, relative, env=None):
        """Import scratch modules without reusing framework-test imports or caches."""
        path = self.root / relative
        script_names = {source.stem for directory in ("scripts", "judge")
                        for source in (self.root / directory).glob("*.py")}
        name = "_atlas_portability_" + path.stem
        with mock.patch.dict(sys.modules), mock.patch.object(sys, "dont_write_bytecode", True), \
                mock.patch.object(sys, "path", [str(self.root / "scripts"), str(self.root / "judge"), str(self.root),
                                              *[entry for entry in sys.path if not str(entry).startswith(str(PACKAGE))]]), \
                mock.patch.dict(os.environ, self.env if env is None else env, clear=True), working_directory(self.caller):
            for key in list(sys.modules):
                if key in script_names or key == "local_life_agent" or key.startswith("local_life_agent."):
                    sys.modules.pop(key, None)
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            yield module

    def main(self, module, args, env=None):
        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch.object(sys, "argv", [module.__file__, *map(str, args)]), \
                mock.patch.dict(os.environ, self.env if env is None else env, clear=True), \
                contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                code = module.main()
            except SystemExit as error:
                code = error.code
                if not isinstance(code, (int, type(None))):
                    stderr.write(str(code))
                    code = 1
        return types.SimpleNamespace(returncode=code or 0, stdout=stdout.getvalue(), stderr=stderr.getvalue())

    def parsed_defaults(self, module, args):
        captured = []
        original = argparse.ArgumentParser.parse_args

        class Parsed(Exception):
            pass

        def capture(parser, *positional, **keywords):
            captured.append(vars(original(parser, *positional, **keywords)))
            raise Parsed

        with mock.patch.object(argparse.ArgumentParser, "parse_args", capture):
            with self.assertRaises(Parsed):
                self.main(module, args)
        self.assertEqual(len(captured), 1)
        return captured[0]

    def populate_legacy(self, root, omit=None):
        for label, folder in LEGACY_RUNS.items():
            if label != omit:
                self.write_jsonl(root / folder / "subset_1200.final.jsonl", self.records)


class RuntimeDefaultsTests(RelocatedCase):
    def test_run_agent_parser_default_without_loading_model_dependencies(self):
        path = self.root / "scripts/run_agent.py"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        # Execute only stdlib imports, constants, and the real argument parser.
        # In particular do not construct a registry or import neural/model code.
        safe = []
        for node in tree.body:
            if isinstance(node, ast.Import) and all(alias.name in {"argparse", "json", "sys", "time", "os"} for alias in node.names):
                safe.append(node)
            elif isinstance(node, ast.ImportFrom) and node.module == "pathlib":
                safe.append(node)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                safe.append(node)
            elif isinstance(node, ast.FunctionDef) and node.name == "parse_args":
                safe.append(node)
        namespace = {"__file__": str(path), "__name__": "_portability_agent_parser"}
        with mock.patch.object(sys, "argv", [str(path)]):
            exec(compile(ast.Module(body=safe, type_ignores=[]), str(path), "exec"), namespace)
            args = namespace["parse_args"]()
        self.assertEqual(Path(args.input), self.root / "query" / AGENT_TASKS)
        self.assertEqual((args.workers, args.max_tool_rounds, args.max_tools_per_round, args.max_tool_calls), (1, 5, 2, 10))

    def test_finalize_and_compare_parser_defaults_follow_relocated_root(self):
        for script, args in (("finalize_run.py", ["--input", self.run]), ("compare_runs.py", ["--run", "fixture=" + str(self.run)])):
            with self.subTest(script=script), self.module("scripts/" + script) as module:
                defaults = self.parsed_defaults(module, args)
                self.assertEqual(Path(defaults["tasks"]), self.root / "query" / AGENT_TASKS)

    def test_finalize_uses_default_tasks_and_creates_nested_output_and_summary(self):
        output = self.caller / "new results" / "deeper" / "final.jsonl"
        rows = [self.records[1], {"task_id": "task-full", "status": "failed", "error": {"type": "FirstFailure"}}, self.records[0], {"task_id": "task-full", "status": "failed", "error": {"type": "LateFailure"}}]
        self.write_jsonl(self.run, rows)
        self.successful(self.command("scripts/finalize_run.py", "--input", self.run, "--output", output))
        final = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(final, self.records)
        summary = json.loads(output.with_name(output.name + ".summary.json").read_text(encoding="utf-8"))
        self.assertEqual((summary["completed"], summary["failed"], summary["tasks_with_multiple_attempts"]), (2, 0, 1))
        self.assertEqual(summary["missing_task_ids"], [])
        self.assertFalse((self.root / "outputs").exists())

    def test_compare_creates_both_independent_nested_parents(self):
        output = self.caller / "tables" / "nested" / "comparison.md"
        dump = self.caller / "summaries" / "other nested" / "comparison.json"
        self.successful(self.command("scripts/compare_runs.py", "--run", "fixture=" + str(self.run), "--output", output, "--dump-summaries", dump))
        self.assertIn("fixture", output.read_text(encoding="utf-8"))
        result = json.loads(dump.read_text(encoding="utf-8"))["fixture"]
        self.assertEqual((result["records"], result["completed"], result["failed"]), (2, 2, 0))
        self.assertFalse(self.judge_artifacts())

    def test_config_paths_and_runtime_budget_invariants_are_preserved(self):
        with self.module("local_life_agent/config.py") as module:
            paths, config = module.PathConfig(), module.AgentConfig()
            self.assertEqual(paths.poi_db, self.root / "data/indexes/poi_rag.sqlite3")
            self.assertEqual(paths.note_db, self.root / "data/indexes/note_rag.sqlite3")
            self.assertEqual(paths.prompt_file, self.root / "prompts/react_system_prompt.md")
            self.assertEqual(paths.output_dir, self.root / "outputs")
            expected = {"max_tool_rounds": 5, "max_tools_per_round": 2, "max_tool_calls": 10,
                        "max_parse_retries": 2, "default_top_k": 10, "max_top_k": 20,
                        "max_observation_chars": 32000, "enable_vector_recall": True,
                        "vector_required": True, "vector_candidate_k": 100, "reranker_batch_size": 4}
            for key, value in expected.items():
                self.assertEqual(getattr(config, key), value, key)
            for kwargs in ({"max_tool_rounds": 6}, {"max_tools_per_round": 3}, {"max_tool_calls": 11}):
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    module.AgentConfig(**kwargs)


class NativeJudgeTests(RelocatedCase):
    def test_taskful_defaults_are_raw_tasks_and_other_judges_have_no_tasks_option(self):
        for dimension, (script, _, _, _) in DIMENSIONS.items():
            with self.subTest(dimension=dimension), self.module(script) as module:
                defaults = self.parsed_defaults(module, ["--run", self.run, "--label", "fixture", "--output", self.temp / "unused", "--dry-run"])
                if dimension in TASKFUL:
                    self.assertEqual(Path(defaults["tasks"]), self.root / "query" / RAW_TASKS)
                else:
                    self.assertNotIn("tasks", defaults)

    def test_four_taskful_and_flex_dry_runs_use_existing_builders_without_key_or_writes(self):
        for dimension, (script, _, _, _) in DIMENSIONS.items():
            if dimension == "factual_accuracy":
                continue
            with self.subTest(dimension=dimension), self.module(script) as module:
                output = self.temp / (dimension + " new output") / "nested" / "score.jsonl"
                before = self.tree(self.root)
                with mock.patch.object(module, "call_judge", side_effect=AssertionError("dry-run called a judge")) as judge:
                    result = self.main(module, ["--run", self.run, "--label", "fixture", "--output", output, "--dry-run", "--fresh"])
                self.successful(result)
                judge.assert_not_called()
                constraints = module.load_constraints(str(self.raw_tasks)) if dimension in TASKFUL else None
                expected = [module.build_payload(record, record["task_id"], constraints.get(record["task_id"], [])) if constraints is not None
                            else module.build_payload(record, record["task_id"]) for record in self.records]
                self.assertEqual(json_stream(result.stdout), expected)
                self.assertEqual([payload["task_id"] for payload in expected], ["task-full", "task-empty"])
                self.assertEqual(expected[0]["query"], self.tasks[0]["query"])
                if dimension in TASKFUL:
                    self.assertEqual([constraint["cid"] for constraint in expected[0]["constraints"]], ["c1", "c2"])
                self.assertFalse(output.parent.exists())
                self.assertEqual(self.tree(self.root), before)

    def test_factual_dry_run_exact_blind_payload_before_credentials_and_output_writes(self):
        with self.module("judge/factual_accuracy_judge.py") as module:
            output = self.temp / "factual no writes" / "nested" / "score.jsonl"
            with mock.patch.object(module, "call_judge", side_effect=AssertionError("factual dry-run called a judge")) as judge:
                result = self.main(module, ["--run", self.run, "--label", "fixture", "--output", output, "--dry-run", "--fresh"])
            self.successful(result)
            judge.assert_not_called()
            payloads = json_stream(result.stdout)
            self.assertEqual(payloads, [self.expected_factual(0), self.expected_factual(1)])
            self.assertNotIn("DO_NOT_EXPOSE", result.stdout)
            self.assertEqual(len(payloads[0]["all_observed_evidence"]), 5)
            self.assertFalse(output.parent.exists())

    def test_factual_dry_run_limit_and_existing_output_are_non_destructive(self):
        output = self.write_jsonl(self.temp / "existing score.jsonl", [{"sentinel": "keep"}])
        before = output.read_bytes()
        with self.module("judge/factual_accuracy_judge.py") as module:
            with mock.patch.object(module, "call_judge", side_effect=AssertionError("unexpected judge call")):
                result = self.main(module, ["--run", self.run, "--label", "fixture", "--output", output, "--dry-run", "--limit", "1", "--fresh"])
        self.successful(result)
        self.assertEqual(json_stream(result.stdout), [self.expected_factual(0)])
        self.assertEqual(output.read_bytes(), before)

    def test_factual_requires_url_when_not_dry_and_run_label_output_even_when_dry(self):
        with self.module("judge/factual_accuracy_judge.py") as module:
            output = self.temp / "never created" / "score.jsonl"
            complete = {"--run": str(self.run), "--label": "fixture", "--output": str(output)}
            for missing in complete:
                args = [item for key, value in complete.items() if key != missing for item in (key, value)] + ["--dry-run"]
                with self.subTest(missing=missing), mock.patch.object(module, "call_judge") as judge:
                    result = self.main(module, args)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(missing, result.stderr)
                    judge.assert_not_called()
            with mock.patch.object(module, "call_judge") as judge:
                result = self.main(module, [item for key, value in complete.items() for item in (key, value)],
                                   env={**self.env, "ATLAS_JUDGE_API_KEY": "fixture-not-a-credential"})
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("--judge-url", result.stderr)
                judge.assert_not_called()
            self.assertFalse(output.parent.exists())

    def test_all_fresh_scoring_outputs_create_parents_using_only_mock_judgments(self):
        for dimension, (script, _, _, score_field) in DIMENSIONS.items():
            with self.subTest(dimension=dimension), self.module(script) as module:
                output = self.temp / "fresh scores" / dimension / "nested" / "score.jsonl"
                payloads = []

                def fake_judge(*args):
                    payloads.append(copy.deepcopy(args[-1]))
                    return {"status": "completed", "judgment": {score_field: 1 if dimension == "factual_accuracy" else 5}, "usage": {}}

                with mock.patch.object(module, "call_judge", side_effect=fake_judge):
                    result = self.main(module, ["--run", self.run, "--label", "fixture", "--output", output,
                                               "--judge-url", "https://example.invalid/not-used", "--judge-api-key-env", "PORTABILITY_TEST_KEY", "--workers", "1", "--fresh"],
                                       env={**self.env, "PORTABILITY_TEST_KEY": "fixture-not-a-credential"})
                self.successful(result)
                rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
                self.assertEqual(len(rows), 2)
                self.assertEqual({row["task_id"] for row in rows}, {"task-full", "task-empty"})
                self.assertTrue(all(row["label"] == "fixture" and row["status"] == "completed" for row in rows))
                if dimension == "factual_accuracy":
                    self.assertEqual(payloads, [self.expected_factual(0), self.expected_factual(1)])


class WrapperTests(RelocatedCase):
    def wrapper(self, dimension, *args, env=None):
        return self.command("scripts/run_all_" + dimension + ".sh", *args, env=env)

    def assert_single_call(self, dimension, output, tasks=None, dry=True):
        calls = self.calls()
        self.assertEqual(len(calls), 1, calls)
        call = calls[0]
        self.assertEqual(self.argument_path(call), self.root / DIMENSIONS[dimension][0])
        self.assertEqual(self.argument_path(call, "--run"), self.run)
        self.assertEqual(self.argument_path(call, "--output"), output)
        self.assertEqual(self.option(call, "--label"), "fixture")
        self.assertEqual("--dry-run" in call["argv"], dry)
        if dimension in TASKFUL:
            self.assertEqual(self.argument_path(call, "--tasks"), tasks or self.raw_tasks)
        else:
            self.assertNotIn("--tasks", call["argv"])
        return call

    def test_help_without_credentials_or_dotenv_invokes_no_judges(self):
        self.assertFalse((self.root / ".env").exists())
        for dimension in DIMENSIONS:
            with self.subTest(dimension=dimension):
                self.clear_calls()
                result = self.wrapper(dimension, "--help")
                self.successful(result)
                flags = ["--run", "--label", "--output", "--runs-root", "--dry-run", "--workers"]
                if dimension in TASKFUL:
                    flags.append("--tasks")
                for flag in flags:
                    self.assertIn(flag, result.stdout)
                if dimension not in TASKFUL:
                    self.assertNotIn("--tasks", result.stdout)
                self.assertEqual(self.calls(), [])
                self.assertFalse(self.judge_artifacts())
                self.assertFalse((self.root / "outputs").exists())

    def test_every_partial_single_mode_fails_before_invocation_or_output_creation(self):
        output = self.caller / "partial mode" / "deep" / "score.jsonl"
        options = [("--run", str(self.run)), ("--label", "fixture"), ("--output", str(output))]
        for dimension in DIMENSIONS:
            for mask in range(1, 7):
                with self.subTest(dimension=dimension, mask=mask):
                    self.clear_calls()
                    args = [item for index, pair in enumerate(options) if mask & (1 << index) for item in pair]
                    result = self.wrapper(dimension, *args, "--dry-run")
                    self.assertNotEqual(result.returncode, 0)
                    self.assertEqual(self.calls(), [])
                    self.assertFalse(output.parent.exists())
                    self.assertFalse(self.judge_artifacts())

    def test_single_dry_run_quotes_interpreter_and_resolves_all_paths_from_caller(self):
        tasks = self.write_json(self.caller / "input data" / "raw tasks with spaces.json", [{"record": task} for task in self.tasks])
        output = self.caller / "new output with spaces" / "nested" / "score.jsonl"
        for dimension in DIMENSIONS:
            with self.subTest(dimension=dimension):
                self.clear_calls()
                args = ["--run", self.run.relative_to(self.caller), "--label", "fixture",
                        "--output", output.relative_to(self.caller)]
                if dimension in TASKFUL:
                    args.extend(("--tasks", tasks.relative_to(self.caller)))
                args.extend(("--workers", "3", "--judge-url", "https://example.invalid/not-used",
                             "--judge-model", "fixture-model", "--judge-api-key-env", "PORTABILITY_TEST_KEY", "--dry-run"))
                self.successful(self.wrapper(dimension, *args))
                call = self.assert_single_call(dimension, output, tasks=tasks)
                self.assertEqual(self.option(call, "--workers"), "3")
                self.assertEqual(self.option(call, "--judge-url"), "https://example.invalid/not-used")
                self.assertEqual(self.option(call, "--judge-model"), "fixture-model")
                self.assertEqual(self.option(call, "--judge-api-key-env"), "PORTABILITY_TEST_KEY")

    def test_single_native_options_forwarded_to_recording_python_not_an_api(self):
        output = self.caller / "native recorded only" / "score.jsonl"
        for dimension in DIMENSIONS:
            with self.subTest(dimension=dimension):
                self.clear_calls()
                self.successful(self.wrapper(dimension, "--run", self.run, "--label", "fixture", "--output", output,
                                             "--workers", "7", "--judge-url", "https://example.invalid/not-used", "--judge-model", "fixture-model",
                                             "--judge-api-key-env", "PORTABILITY_TEST_KEY",
                                             env={**self.env, "PORTABILITY_TEST_KEY": "fixture-not-a-credential"}))
                call = self.assert_single_call(dimension, output, dry=False)
                self.assertEqual(self.option(call, "--workers"), "7")
                self.assertEqual(self.option(call, "--judge-url"), "https://example.invalid/not-used")
                self.assertEqual(self.option(call, "--judge-model"), "fixture-model")
                self.assertEqual(self.option(call, "--judge-api-key-env"), "PORTABILITY_TEST_KEY")
                self.assertFalse(output.exists(), "the recording interpreter must not score")

    def test_python_defaults_to_python3_from_path(self):
        fallback = self.caller / "fallback interpreter directory" / "python3"
        self.make_fake_python(fallback)
        env = {key: value for key, value in self.env.items() if key != "PYTHON"}
        env["PATH"] = str(fallback.parent) + os.pathsep + os.defpath
        output = self.caller / "fallback output" / "score.jsonl"
        for dimension in DIMENSIONS:
            with self.subTest(dimension=dimension):
                self.clear_calls()
                self.successful(self.wrapper(dimension, "--run", self.run, "--label", "fixture", "--output", output, "--dry-run", env=env))
                self.assert_single_call(dimension, output)

    def assert_legacy_calls(self, dimension, runs_root, smoke=False):
        calls = self.calls()
        self.assertEqual(len(calls), 14)
        self.assertEqual([self.option(call, "--label") for call in calls], list(LEGACY_RUNS))
        script, output_folder, workers, _ = DIMENSIONS[dimension]
        for call, (label, folder) in zip(calls, LEGACY_RUNS.items()):
            self.assertEqual(self.argument_path(call), self.root / script)
            self.assertEqual(self.argument_path(call, "--run"), runs_root / folder / "subset_1200.final.jsonl")
            base = self.root / "judge" / output_folder
            self.assertEqual(self.argument_path(call, "--output"), (base / "smoke" if smoke else base) / (label + ".jsonl"))
            self.assertEqual(self.option(call, "--workers"), "5" if smoke else str(workers))
            if smoke:
                self.assertEqual(self.option(call, "--limit"), "15")
            else:
                self.assertNotIn("--limit", call["argv"])
            if dimension in TASKFUL:
                self.assertEqual(self.argument_path(call, "--tasks"), self.raw_tasks)
            else:
                self.assertNotIn("--tasks", call["argv"])

    def test_legacy_batch_default_runs_root_preserves_all_fourteen_exact_maps(self):
        runs_root = self.root / "outputs"
        self.populate_legacy(runs_root)
        for dimension in DIMENSIONS:
            with self.subTest(dimension=dimension):
                self.clear_calls()
                self.successful(self.wrapper(dimension, "--dry-run"))
                self.assert_legacy_calls(dimension, runs_root)
                self.assertTrue(all("--dry-run" in call["argv"] for call in self.calls()))

    def test_batch_relative_runs_root_with_spaces_resolves_before_cd(self):
        runs_root = self.caller / "external runs with spaces"
        self.populate_legacy(runs_root)
        for dimension in DIMENSIONS:
            with self.subTest(dimension=dimension):
                self.clear_calls()
                self.successful(self.wrapper(dimension, "--runs-root", runs_root.relative_to(self.caller), "--dry-run"))
                self.assert_legacy_calls(dimension, runs_root)
                self.assertFalse((self.root / "outputs").exists())

    def test_original_five_smoke_modes_preserve_limit_and_workers(self):
        runs_root = self.caller / "smoke runs"
        self.populate_legacy(runs_root)
        # Factual accuracy had no original smoke mode; do not invent its semantics.
        for dimension in DIMENSIONS.keys() - {"factual_accuracy"}:
            with self.subTest(dimension=dimension):
                self.clear_calls()
                self.successful(self.wrapper(dimension, "smoke", "--runs-root", runs_root, "--dry-run"))
                self.assert_legacy_calls(dimension, runs_root, smoke=True)

    def test_missing_last_batch_run_prefails_without_any_judge_or_new_output_directory(self):
        runs_root = self.caller / "incomplete runs"
        self.populate_legacy(runs_root, omit="gemini38_flash")
        before_root, before_caller = self.tree(self.root), self.tree(self.caller)
        for dimension in DIMENSIONS:
            for dry in (True, False):
                with self.subTest(dimension=dimension, dry=dry):
                    self.clear_calls()
                    args = ["--runs-root", runs_root.relative_to(self.caller)]
                    if dry:
                        args.append("--dry-run")
                    result = self.wrapper(dimension, *args,
                                          env={**self.env, "ATLAS_JUDGE_API_KEY": "fixture-not-a-credential",
                                               "ATLAS_JUDGE_URL": "https://example.invalid/not-used"})
                    self.assertNotEqual(result.returncode, 0)
                    self.assertEqual(self.calls(), [], "all runs must be preflighted before even the first judge")
                    self.assertFalse(self.judge_artifacts())
                    self.assertFalse((self.root / "outputs").exists())
                    self.assertEqual(self.tree(self.root), before_root)
                    self.assertEqual(self.tree(self.caller), before_caller)


class QoderBuilderTests(RelocatedCase):
    def builder(self, *args):
        return self.command("judge/build_qoder_judge_inputs.py", *args)

    def generated_payloads(self, judge_dir, label="fixture"):
        return [payload for chunk in sorted((judge_dir / label / "inputs").glob("chunk_*.json"))
                for payload in json.loads(chunk.read_text(encoding="utf-8"))]

    def test_single_custom_label_raw_wrappers_schema_and_payload_match_original_builder(self):
        judge_dir = self.temp / "new judge tree" / "nested"
        before = self.run.read_bytes()
        self.successful(self.builder("--run", self.run, "--label", "fixture", "--tasks", self.raw_tasks, "--judge-dir", judge_dir))
        expected = [self.expected_qoder(0), self.expected_qoder(1)]
        self.assertEqual(self.generated_payloads(judge_dir), expected)
        with self.module("judge/build_qoder_judge_inputs.py") as module:
            self.assertEqual(module.MODELS, LEGACY_RUNS)
            tasks = [{**task, "satisfiable": task["answer"]["satisfiable"]} for task in self.tasks]
            self.assertEqual([module.build_payload(task, record) for task, record in zip(tasks, self.records)], expected)
        self.assertNotIn("DO_NOT_EXPOSE", json.dumps(expected, ensure_ascii=False))
        self.assertEqual(self.run.read_bytes(), before)
        self.assertFalse(self.judge_artifacts())

    def test_budget_option_chunks_both_tasks_and_preserves_existing_verdicts(self):
        judge_dir = self.temp / "provided judge directory"
        verdict = self.write_jsonl(judge_dir / "fixture" / "verdicts" / "chunk_01.jsonl", [{"sentinel": "existing verdict"}])
        before = verdict.read_bytes()
        self.successful(self.builder("--run", self.run, "--label", "fixture", "--tasks", self.raw_tasks, "--judge-dir", judge_dir, "--budget-chars", "1"))
        self.assertEqual(self.generated_payloads(judge_dir), [self.expected_qoder(0), self.expected_qoder(1)])
        self.assertEqual(len(list((judge_dir / "fixture/inputs").glob("chunk_*.json"))), 2)
        self.assertEqual(verdict.read_bytes(), before)
        manifest = json.loads((judge_dir / "_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["budget_chars"], 1)
        self.assertEqual(len(manifest["models"]), 1)
        self.assertEqual((manifest["models"][0]["queries"], manifest["models"][0]["recommendations"]), (2, 3))

    def test_default_tasks_resolve_inside_relocated_package(self):
        judge_dir = self.temp / "default tasks judge"
        self.successful(self.builder("--run", self.run, "--label", "fixture", "--judge-dir", judge_dir))
        self.assertEqual(self.generated_payloads(judge_dir), [self.expected_qoder(0), self.expected_qoder(1)])

    def test_legacy_positional_budget_and_model_selector_use_supplied_runs_root(self):
        runs_root = self.caller / "legacy runs with spaces"
        label = "qwen35_4b_nothink"
        self.write_jsonl(runs_root / LEGACY_RUNS[label] / "subset_1200.final.jsonl", self.records)
        judge_dir = self.temp / "legacy judge"
        self.successful(self.builder("--runs-root", runs_root, "--tasks", self.raw_tasks, "--judge-dir", judge_dir, "1", label))
        self.assertEqual(self.generated_payloads(judge_dir, label), [self.expected_qoder(0), self.expected_qoder(1)])
        self.assertEqual({path.name for path in judge_dir.iterdir() if path.is_dir()}, {label})

    def test_single_mode_requires_label_and_rejects_path_traversal_before_writes(self):
        judge_dir = self.temp / "invalid label output"
        # Even a broken implementation must keep traversal attempts in scratch.
        absolute_label = self.temp / "absolute label output"
        for label in (None, "../escape", str(absolute_label), "nested/label", ".", ".."):
            with self.subTest(label=label):
                args = ["--run", self.run, "--tasks", self.raw_tasks, "--judge-dir", judge_dir]
                if label is not None:
                    args.extend(("--label", label))
                result = self.builder(*args)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(judge_dir.exists())
                self.assertFalse(self.judge_artifacts())
                self.assertFalse((self.temp / "escape").exists())
                self.assertFalse(absolute_label.exists())
        result = self.builder("--label", "fixture", "--tasks", self.raw_tasks, "--judge-dir", judge_dir)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(judge_dir.exists())

    def test_bad_run_sources_fail_before_creating_any_output(self):
        unknown = {**self.records[0], "task_id": "not-in-tasks"}
        cases = {"unknown-id": [unknown], "empty": [], "no-completed": [{"task_id": "task-full", "status": "failed"}]}
        for name, rows in cases.items():
            self.write_jsonl(self.caller / (name + ".jsonl"), rows)
        (self.caller / "malformed.jsonl").write_text('{"task_id": broken\n', encoding="utf-8")
        for name in (*cases, "malformed"):
            with self.subTest(name=name):
                judge_dir = self.temp / ("bad source output " + name)
                result = self.builder("--run", self.caller / (name + ".jsonl"), "--label", "fixture", "--tasks", self.raw_tasks, "--judge-dir", judge_dir)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(judge_dir.exists())
                self.assertFalse(self.judge_artifacts())

    def test_malformed_or_empty_task_source_fails_without_empty_manifest(self):
        for name, source in (("malformed", {"wrong": "schema"}), ("empty", [])):
            with self.subTest(name=name):
                tasks = self.write_json(self.caller / (name + " tasks.json"), source)
                judge_dir = self.temp / (name + " tasks output")
                result = self.builder("--run", self.run, "--label", "fixture", "--tasks", tasks, "--judge-dir", judge_dir)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(judge_dir.exists())


class QoderSummaryTests(RelocatedCase):
    def summary_fixture(self, judge_dir, label="fixture", incomplete=False):
        self.write_json(judge_dir / label / "inputs/chunk_01.json", [self.expected_qoder(0), self.expected_qoder(1)])
        rows = [{"task_id": "task-full", "rank": rank, "verdict": verdict, "constraints_met": met, "reason": "fixture"}
                for rank, verdict, met in ((1, "qualified", 2), (2, "acceptable", 1), (3, "unacceptable", 0))]
        self.write_jsonl(judge_dir / label / "verdicts/chunk_01.jsonl", rows[:2] if incomplete else rows)

    def summarize(self, judge_dir, label):
        return self.command("judge/summarize_qoder_judge.py", "--judge-dir", judge_dir, label)

    def test_custom_label_keeps_qualified_only_nonempty_and_legacy_cm_calculation(self):
        judge_dir = self.temp / "summary custom root"
        self.summary_fixture(judge_dir)
        self.successful(self.summarize(judge_dir, "fixture"))
        summary = json.loads((judge_dir / "fixture/summary.json").read_text(encoding="utf-8"))
        expected = {
            "model": "fixture", "label": "fixture", "expected_recommendations": 3, "judged_recommendations": 3,
            "nonempty_queries": 1, "incomplete_tasks": 0, "incomplete_task_ids": [],
            "verdict_counts": {"qualified": 1, "acceptable": 1, "unacceptable": 1},
            "qualified_pois_per_query": 1.0, "macro_qualified_ratio": 0.3333,
            "micro_qualified_ratio": 0.3333, "success_at_k": 1.0,
            # Legacy calculation: (2 + 1 + 0) / (2 + 2 + 2), not paper CSR.
            "constraint_satisfaction_rate": 0.5,
            "avg_constraints_met_acceptable": 1.0, "avg_constraints_met_unacceptable": 0.0,
        }
        for key, value in expected.items():
            self.assertEqual(summary[key], value, key)
        self.assertEqual(json.loads((judge_dir / "_overall.json").read_text(encoding="utf-8")), [summary])
        self.assertIn("fixture", (judge_dir / "_overall.md").read_text(encoding="utf-8"))
        self.assertFalse(self.judge_artifacts())

    def test_incomplete_nonempty_task_report_does_not_count_zero_recommendation_task(self):
        judge_dir = self.temp / "incomplete summary"
        self.summary_fixture(judge_dir, incomplete=True)
        self.successful(self.summarize(judge_dir, "fixture"))
        result = json.loads((judge_dir / "fixture/summary.json").read_text(encoding="utf-8"))
        self.assertEqual((result["expected_recommendations"], result["judged_recommendations"], result["nonempty_queries"]), (3, 2, 1))
        self.assertEqual(result["incomplete_task_ids"], ["task-full"])
        self.assertEqual(result["incomplete_tasks"], 1)
        self.assertEqual(result["micro_qualified_ratio"], 0.5)
        self.assertEqual(result["constraint_satisfaction_rate"], 0.75)

    def test_legacy_model_label_still_works_under_supplied_judge_dir(self):
        judge_dir = self.temp / "legacy summary root"
        label = "qwen35_4b_nothink"
        self.summary_fixture(judge_dir, label)
        self.successful(self.summarize(judge_dir, label))
        result = json.loads((judge_dir / label / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(result["label"], "Qwen3.5-4B w/o Think")
        self.assertEqual(result["micro_qualified_ratio"], 0.3333)
        self.assertEqual(result["nonempty_queries"], 1)

    def test_explicit_missing_label_fails_without_creating_empty_overall_reports(self):
        for existing in (False, True):
            with self.subTest(existing_root=existing):
                judge_dir = self.temp / ("missing label root " + str(existing))
                if existing:
                    self.summary_fixture(judge_dir)
                before = self.tree(judge_dir)
                result = self.summarize(judge_dir, "missing-label")
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(self.tree(judge_dir), before)
                self.assertFalse((judge_dir / "_overall.json").exists())
                self.assertFalse((judge_dir / "_overall.md").exists())
                self.assertFalse(self.judge_artifacts())


class ProfileTests(RelocatedCase):
    def test_relocated_root_input_and_default_external_interpreter(self):
        env = {key: value for key, value in self.env.items() if key != "PYTHON"}
        with self.module(PROFILE_REL, env=env) as module:
            self.assertEqual(module.ROOT, self.root)
            self.assertEqual(module.DEFAULT_INPUT, self.root / "query" / AGENT_TASKS)
            self.assertEqual(Path(module.PYTHON), Path(sys.executable).resolve())
            self.assertEqual(module.PROFILES, PROFILES)
            args = self.parsed_defaults(module, ["qwen35-local-no-think", "smoke"])
            self.assertEqual(Path(args["input"]), self.agent_tasks)

    def test_absolute_and_relative_python_with_spaces_resolve_from_caller(self):
        for value in (str(self.fake_python), str(self.fake_python.relative_to(self.caller))):
            env = {**self.env, "PYTHON": value}
            with self.subTest(value=value), self.module(PROFILE_REL, env=env) as module:
                self.assertEqual(Path(module.PYTHON), self.fake_python)
                with mock.patch.object(module.subprocess, "run", return_value=types.SimpleNamespace(returncode=0)) as run:
                    self.successful(self.main(module, ["qwen35-local-no-think", "finalize"], env=env))
                self.assertEqual(Path(run.call_args.args[0][0]), self.fake_python)

    def test_all_profiles_smoke_full_retry_commands_and_generation_settings_are_unchanged(self):
        env = {**self.env, "ATLAS_HOSTED_API_KEY": "fixture-not-a-credential",
               "ATLAS_HOSTED_API_BASE": "https://example.invalid/not-used",
               "LOCAL_LIFE_MODEL": "discard-stale-model", "LOCAL_LIFE_REASONING_EFFORT": "discard-stale-effort",
               "LOCAL_LIFE_OMIT_TEMPERATURE": "true", "LOCAL_LIFE_CACHE_SYSTEM_PROMPT": "true"}
        with self.module(PROFILE_REL, env=env) as module:
            for name, profile in PROFILES.items():
                for stage in ("smoke", "full", "retry"):
                    with self.subTest(profile=name, stage=stage), \
                            mock.patch.object(module.subprocess, "run", return_value=types.SimpleNamespace(returncode=0)) as run:
                        self.successful(self.main(module, [name, stage], env=env))
                        command = run.call_args.args[0]
                        kwargs = run.call_args.kwargs
                        self.assertEqual(Path(command[0]), self.fake_python)
                        self.assertEqual(Path(command[1]), self.root / "scripts/run_agent.py")
                        self.assertEqual(Path(self.option(command, "--input")), self.agent_tasks)
                        directory = self.root / "outputs" / (("_smoke_" if stage == "smoke" else "") + profile["slug"])
                        self.assertEqual(Path(self.option(command, "--output")), directory / ("smoke.jsonl" if stage == "smoke" else "subset_1200.jsonl"))
                        workers = min(4, profile["workers"]) if stage == "smoke" else profile["retry_workers"] if stage == "retry" else profile["workers"]
                        self.assertEqual(self.option(command, "--workers"), str(workers))
                        self.assertEqual("--limit" in command, stage == "smoke")
                        if stage == "smoke":
                            self.assertEqual(self.option(command, "--limit"), "10")
                        self.assertEqual(kwargs["cwd"], self.root)
                        generated = kwargs["env"]
                        self.assertEqual(generated["LOCAL_LIFE_MODEL"], profile["model"])
                        budget = profile.get("retry_output", profile["output"]) if stage == "retry" else profile["output"]
                        self.assertEqual(generated["LOCAL_LIFE_MAX_OUTPUT_TOKENS"], str(budget))
                        self.assertEqual(generated["LOCAL_LIFE_API_RETRIES"], str(profile.get("api_retries", 3)))
                        self.assertEqual(generated["LOCAL_LIFE_API_TIMEOUT"], "900")
                        self.assertEqual(generated["LOCAL_LIFE_API_STYLE"], "chat_completions")
                        if profile["provider"] == "local":
                            self.assertEqual(generated["LOCAL_LIFE_API_BASE"], "http://127.0.0.1:8001/v1")
                            self.assertEqual(generated["LOCAL_LIFE_ENABLE_THINKING"], str(profile["thinking"]).lower())
                            self.assertEqual(generated["LOCAL_LIFE_THINKING_FIELD"], "chat_template_kwargs")
                        else:
                            self.assertNotIn("LOCAL_LIFE_ENABLE_THINKING", generated)
                        if profile.get("omit_temperature"):
                            self.assertEqual(generated["LOCAL_LIFE_OMIT_TEMPERATURE"], "true")
                            self.assertNotIn("LOCAL_LIFE_TEMPERATURE", generated)
                        else:
                            self.assertEqual(generated["LOCAL_LIFE_TEMPERATURE"], "0.1")
                            self.assertNotIn("LOCAL_LIFE_OMIT_TEMPERATURE", generated)
                        for field, env_key in (("reasoning_effort", "LOCAL_LIFE_REASONING_EFFORT"), ("repetition_penalty", "LOCAL_LIFE_REPETITION_PENALTY")):
                            self.assertEqual(generated.get(env_key), str(profile[field]) if field in profile else None)
                        self.assertEqual(generated.get("LOCAL_LIFE_CACHE_SYSTEM_PROMPT"), "true" if profile.get("cache_system_prompt") else None)
                        self.assertEqual(generated["LOCAL_LIFE_VECTOR_DEVICE"], "cuda")
                        self.assertEqual(generated["LOCAL_LIFE_RERANKER_DEVICE"], "cuda")

    def test_endpoint_credentials_are_read_at_call_time_and_local_endpoint_is_overridable(self):
        with self.module(PROFILE_REL) as module:
            hosted = {**self.env, "ATLAS_HOSTED_API_KEY": "fixture-not-a-credential",
                      "ATLAS_HOSTED_API_BASE": "https://example.invalid/v1"}
            with mock.patch.object(module.subprocess, "run",
                                   return_value=types.SimpleNamespace(returncode=0)) as run:
                self.successful(self.main(module, ["glm53", "smoke"], env=hosted))
            generated = run.call_args.kwargs["env"]
            self.assertEqual(generated["LOCAL_LIFE_API_BASE"], "https://example.invalid/v1")
            self.assertEqual(generated["LOCAL_LIFE_API_KEY"], "fixture-not-a-credential")

            local = {**self.env, "ATLAS_LOCAL_API_BASE": "http://127.0.0.1:9000/v1",
                     "ATLAS_LOCAL_API_KEY": "fixture-local-key"}
            with mock.patch.object(module.subprocess, "run",
                                   return_value=types.SimpleNamespace(returncode=0)) as run:
                self.successful(self.main(module, ["qwen35-local-no-think", "smoke"], env=local))
            generated = run.call_args.kwargs["env"]
            self.assertEqual(generated["LOCAL_LIFE_API_BASE"], "http://127.0.0.1:9000/v1")
            self.assertEqual(generated["LOCAL_LIFE_API_KEY"], "fixture-local-key")
            self.assertEqual(generated["LOCAL_LIFE_ENABLE_THINKING"], "false")
            self.assertEqual(generated["LOCAL_LIFE_THINKING_FIELD"], "chat_template_kwargs")

    def test_every_finalize_command_is_mocked_and_preserves_profile_config(self):
        with self.module(PROFILE_REL) as module:
            for name, profile in PROFILES.items():
                with self.subTest(profile=name), mock.patch.object(module.subprocess, "run", return_value=types.SimpleNamespace(returncode=0)) as run:
                    self.successful(self.main(module, [name, "finalize"]))
                    command = run.call_args.args[0]
                    directory = self.root / "outputs" / profile["slug"]
                    self.assertEqual(Path(command[0]), self.fake_python)
                    self.assertEqual(Path(command[1]), self.root / "scripts/finalize_run.py")
                    self.assertEqual(Path(self.option(command, "--input")), directory / "subset_1200.jsonl")
                    self.assertEqual(Path(self.option(command, "--output")), directory / "subset_1200.final.jsonl")
                    self.assertEqual(Path(self.option(command, "--tasks")), self.agent_tasks)
                    self.assertEqual(json.loads(self.option(command, "--config-json")), {**profile, "profile": name})
                    self.assertEqual(run.call_args.kwargs["cwd"], self.root)
            self.assertFalse((self.root / "outputs").exists(), "finalize was mocked, not executed")

    def test_generation_overrides_and_fresh_are_forwarded_without_changing_profiles(self):
        with self.module(PROFILE_REL) as module:
            before = copy.deepcopy(module.PROFILES)
            with mock.patch.object(module.subprocess, "run", return_value=types.SimpleNamespace(returncode=0)) as run:
                self.successful(self.main(module, ["qwen35-local-think", "retry", "--workers", "7", "--max-output-tokens", "1234", "--reasoning-effort", "medium", "--fresh"]))
            command = run.call_args.args[0]
            self.assertEqual(self.option(command, "--workers"), "7")
            self.assertIn("--fresh", command)
            self.assertEqual(run.call_args.kwargs["env"]["LOCAL_LIFE_MAX_OUTPUT_TOKENS"], "1234")
            self.assertEqual(run.call_args.kwargs["env"]["LOCAL_LIFE_REASONING_EFFORT"], "medium")
            self.assertEqual(module.PROFILES, before)

    def test_local_profile_dry_run_only_creates_scratch_outputs_and_never_runs_agent(self):
        env = {key: value for key, value in self.env.items() if key != "PYTHON"}
        for stage in ("smoke", "full", "retry"):
            with self.subTest(stage=stage):
                result = self.command(PROFILE_REL, "qwen35-local-no-think", stage, "--dry-run", env=env)
                self.successful(result)
                payload = json.loads(result.stdout)
                self.assertEqual(Path(payload["command"][0]), Path(sys.executable).resolve())
                self.assertEqual(Path(self.option(payload["command"], "--input")), self.agent_tasks)
                self.assertNotIn("LOCAL_LIFE_API_KEY", payload["env"])
                output = Path(self.option(payload["command"], "--output"))
                self.assertTrue(output.parent.is_dir())
                self.assertFalse(output.exists())
                self.assertEqual(self.calls(), [])
        self.assertFalse((self.root / "data").exists())
        self.assertFalse((self.root / ".venv").exists())
        self.assertFalse((self.root / ".env").exists())

    def test_status_and_list_profiles_do_not_run_any_process_or_require_credentials(self):
        with self.module(PROFILE_REL) as module, mock.patch.object(module.subprocess, "run") as run:
            listed = self.main(module, ["--list-profiles"])
            self.successful(listed)
            self.assertEqual(listed.stdout.splitlines(), sorted(PROFILES))
            result = self.main(module, ["qwen35-local-no-think", "status"])
            self.successful(result)
            status = json.loads(result.stdout)
            self.assertEqual((status["records"], status["completed"], status["failed"]), (0, 0, 0))
            run.assert_not_called()
            self.assertFalse((self.root / "outputs").exists())


if __name__ == "__main__":
    unittest.main()
