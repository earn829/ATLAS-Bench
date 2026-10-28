"""Offline contract tests for the atlas-qoder-audit skill.

Synthetic two-chunk fixtures only: no model, no network, no repository judge data.
Expected metric values are hand-computed from the published formulas in
references/verdict-contract.md, not read back from the implementation.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PACKAGE = Path(__file__).resolve().parents[1]
SKILL = PACKAGE / ".qoder/skills/atlas-qoder-audit"
STATUS = SKILL / "scripts/audit_status.py"
SCORER = SKILL / "scripts/score_verdicts.py"

PAYLOADS = [
    {"task_id": "t1", "query": "找家安静的咖啡馆", "context": {"time": "2025-05-01 10:00"},
     "constraints": [{"cid": "c1", "h": "刚性"}, {"cid": "c2", "h": "柔性"}], "n_constraints": 2,
     "satisfiable": True,
     "recommendations": [{"rank": 1, "poiid": "P1", "name": "甲"}, {"rank": 2, "poiid": "P2", "name": "乙"}]},
    {"task_id": "t2", "query": "有没有符合条件的店", "context": {},
     "constraints": [{"cid": "c1", "h": "刚性"}], "n_constraints": 1, "satisfiable": False,
     "recommendations": []},
    {"task_id": "t3", "query": "附近川菜", "context": {},
     "constraints": [{"cid": "c1", "h": "刚性"}], "n_constraints": 1, "satisfiable": True,
     "recommendations": [{"rank": 1, "poiid": "P3", "name": "丙"}]},
]
GOOD_ROWS = [
    {"task_id": "t1", "rank": 1, "poiid": "P1", "verdict": "qualified", "constraints_met": 2, "reason": "类别与安静均有自身证据"},
    {"task_id": "t1", "rank": 2, "poiid": "P2", "verdict": "unacceptable", "constraints_met": 1, "reason": "刚性类别无证据"},
    {"task_id": "t3", "rank": 1, "poiid": "P3", "verdict": "acceptable", "constraints_met": 0, "reason": "软偏好已披露缺口"},
]


def write_fixture(judge_dir: Path, label: str, rows, chunks=((0, 2), (2, 3))) -> Path:
    label_dir = judge_dir / label
    (label_dir / "inputs").mkdir(parents=True, exist_ok=True)
    (label_dir / "verdicts").mkdir(parents=True, exist_ok=True)
    for index, (start, stop) in enumerate(chunks, 1):
        (label_dir / "inputs" / f"chunk_{index:02d}.json").write_text(
            json.dumps(PAYLOADS[start:stop], ensure_ascii=False), encoding="utf-8")
    by_chunk = {index: [] for index in range(1, len(chunks) + 1)}
    task_chunk = {}
    for index, (start, stop) in enumerate(chunks, 1):
        for payload in PAYLOADS[start:stop]:
            task_chunk[payload["task_id"]] = index
    for row in rows:
        by_chunk[task_chunk.get(row["task_id"], row.get("_chunk", 1))].append(row)
    for index, chunk_rows in by_chunk.items():
        if chunk_rows:
            (label_dir / "verdicts" / f"chunk_{index:02d}.jsonl").write_text(
                "".join(json.dumps({k: v for k, v in r.items() if k != "_chunk"}, ensure_ascii=False) + "\n"
                        for r in chunk_rows), encoding="utf-8")
    return label_dir


class SkillScriptCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="atlas-audit-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.caller = self.tmp / "caller"
        self.judge = self.caller / "judge"
        self.caller.mkdir(parents=True)

    def run_script(self, script: Path, *args, cwd=None):
        return subprocess.run([sys.executable, str(script), *map(str, args)],
                              cwd=str(cwd or self.caller), text=True, capture_output=True, timeout=60)

    def score(self, *args, cwd=None):
        return self.run_script(SCORER, "--judge-dir", "judge" if cwd is None else self.judge, *args, cwd=cwd)


class TestPublishedScoring(SkillScriptCase):
    def test_complete_verdicts_reproduce_hand_computed_metrics(self):
        write_fixture(self.judge, "arm_a", GOOD_ROWS)
        result = self.score("arm_a")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        summary = json.loads((self.judge / "arm_a" / "summary_published.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["queries"], 3)
        self.assertEqual(summary["queries_in_inputs"], 3)
        self.assertEqual(summary["should_answer_queries"], 2)
        self.assertEqual(summary["unsatisfiable_queries"], 1)
        self.assertEqual(summary["judged_rows"], 3)
        self.assertEqual(summary["expected_rows"], 3)
        self.assertEqual(summary["coverage"], 1.0)
        self.assertEqual(summary["constraint_denominator"], 5)
        self.assertEqual(summary["csr"], 40.0)
        self.assertEqual(summary["cr"], 60.0)
        self.assertEqual(summary["success_at_k"], 100.0)
        self.assertEqual(summary["recs_per_query"], 1.5)
        self.assertEqual(summary["verdict_counts"], {"qualified": 1, "acceptable": 1, "unacceptable": 1})
        self.assertEqual(summary["avg_constraints_met_by_verdict"],
                         {"qualified": 2.0, "acceptable": 0.0, "unacceptable": 1.0})
        self.assertEqual(summary["avg_constraints_per_query"], 1.3333)
        table = (self.judge / "_published_metrics.md").read_text(encoding="utf-8")
        self.assertIn("| arm_a | 3/3 | 2 | 3/3 | 1.000 | 40.00 | 60.00 | 100.00 | 1.50 |", table)

    def test_no_write_leaves_judge_dir_untouched(self):
        label_dir = write_fixture(self.judge, "arm_a", GOOD_ROWS)
        before = {path: path.read_bytes() for path in sorted(label_dir.rglob("*")) if path.is_file()}
        result = self.run_script(SCORER, "--judge-dir", str(self.judge), "--no-write", "--json", "arm_a")
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["labels"]["arm_a"]["csr"], 40.0)
        after = {path: path.read_bytes() for path in sorted(label_dir.rglob("*")) if path.is_file()}
        self.assertEqual(before, after)
        self.assertFalse((self.judge / "_published_metrics.md").exists())

    def test_partial_coverage_scores_only_complete_queries(self):
        write_fixture(self.judge, "arm_a", GOOD_ROWS[:1])
        result = self.score("arm_a")
        self.assertEqual(result.returncode, 0, result.stderr)
        summary = json.loads((self.judge / "arm_a" / "summary_published.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["queries"], 1)
        self.assertEqual(summary["should_answer_queries"], 0)
        self.assertEqual(summary["written_rows"], 1)
        self.assertEqual(summary["judged_rows"], 0)
        self.assertEqual(summary["coverage"], 0.0)
        self.assertIsNone(summary["csr"])
        self.assertIn("incomplete coverage", result.stderr)

    def test_relative_judge_dir_resolves_against_caller_directory(self):
        write_fixture(self.judge, "arm_a", GOOD_ROWS)
        result = self.score("arm_a")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.judge / "arm_a" / "summary_published.json").is_file())

    def test_several_labels_are_scored_in_one_table(self):
        write_fixture(self.judge, "arm_a", GOOD_ROWS)
        write_fixture(self.judge, "arm_b", GOOD_ROWS)
        result = self.run_script(SCORER, "--judge-dir", str(self.judge), "--no-write")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("arm_a", result.stdout)
        self.assertIn("arm_b", result.stdout)

    def test_every_label_is_discovered_when_none_is_named(self):
        write_fixture(self.judge, "arm_a", GOOD_ROWS)
        write_fixture(self.judge, "arm_b", GOOD_ROWS)
        result = self.run_script(SCORER, "--judge-dir", str(self.judge), "--no-write")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count("| arm_"), 2)


class TestContractViolations(SkillScriptCase):
    def assert_rejected(self, rows, fragment):
        write_fixture(self.judge, "arm_a", rows)
        result = self.score("arm_a")
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn(fragment, result.stderr)
        self.assertFalse((self.judge / "arm_a" / "summary_published.json").exists())

    def broken(self, **changes):
        row = dict(GOOD_ROWS[0], **changes)
        return [row, *GOOD_ROWS[1:]]

    def test_qualified_below_n_constraints_is_rejected(self):
        self.assert_rejected(self.broken(constraints_met=1), "qualified requires constraints_met == n_constraints (2)")

    def test_poiid_mismatch_is_rejected(self):
        self.assert_rejected(self.broken(poiid="PX"), "poiid 'PX' != input 'P1'")

    def test_unknown_verdict_is_rejected(self):
        self.assert_rejected(self.broken(verdict="great"), "verdict must be one of")

    def test_out_of_range_constraints_met_is_rejected(self):
        self.assert_rejected(self.broken(verdict="unacceptable", constraints_met=5), "outside 0..2")

    def test_empty_reason_is_rejected(self):
        self.assert_rejected(self.broken(reason="  "), "reason must be a non-empty string")

    def test_unknown_task_is_rejected(self):
        self.assert_rejected([*GOOD_ROWS, {"task_id": "t9", "rank": 1, "poiid": "P9", "verdict": "qualified",
                                           "constraints_met": 1, "reason": "x", "_chunk": 1}],
                             "task_id absent from the blind inputs")

    def test_duplicate_row_is_rejected(self):
        self.assert_rejected([*GOOD_ROWS, dict(GOOD_ROWS[0])], "duplicate verdict row")

    def test_missing_rank_is_rejected(self):
        self.assert_rejected([dict(GOOD_ROWS[0], rank=7), *GOOD_ROWS[1:]], "rank not among the input recommendations")

    def test_row_in_wrong_chunk_is_rejected(self):
        label_dir = write_fixture(self.judge, "arm_a", GOOD_ROWS)
        rows = (label_dir / "verdicts" / "chunk_01.jsonl").read_text(encoding="utf-8")
        (label_dir / "verdicts" / "chunk_01.jsonl").unlink()
        (label_dir / "verdicts" / "chunk_02.jsonl").write_text(rows, encoding="utf-8")
        result = self.score("arm_a")
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("verdict written to chunk_02 but the query lives in chunk_01", result.stderr)

    def test_missing_inputs_explain_the_next_step(self):
        (self.judge / "arm_a" / "verdicts").mkdir(parents=True)
        result = self.score("arm_a")
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("build_qoder_judge_inputs.py", result.stderr)


class TestAuditStatus(SkillScriptCase):
    def test_complete_label_reports_full_coverage(self):
        write_fixture(self.judge, "arm_a", GOOD_ROWS)
        result = self.run_script(STATUS, "--judge-dir", str(self.judge), "arm_a")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("COMPLETE  chunks 2/2  queries 3  rows 3/3", result.stdout)
        self.assertNotIn("pending", result.stdout)

    def test_pending_worklist_names_unfinished_chunks(self):
        write_fixture(self.judge, "arm_a", GOOD_ROWS[:1])
        result = self.run_script(STATUS, "--judge-dir", str(self.judge), "--pending", "arm_a")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.split(), ["chunk_01", "chunk_02"])

    def test_anomalies_are_listed_and_keep_the_chunk_pending(self):
        write_fixture(self.judge, "arm_a", [dict(GOOD_ROWS[0], verdict="great"), *GOOD_ROWS[1:]])
        result = self.run_script(STATUS, "--judge-dir", str(self.judge), "arm_a")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("IN PROGRESS", result.stdout)
        self.assertIn("bad verdict 'great'", result.stdout)
        pending = self.run_script(STATUS, "--judge-dir", str(self.judge), "--pending", "arm_a")
        self.assertEqual(pending.stdout.split(), ["chunk_01"])

    def test_status_flags_poiid_and_constraints_met_violations(self):
        write_fixture(self.judge, "arm_a", [dict(GOOD_ROWS[0], poiid="PX"),
                                            dict(GOOD_ROWS[1], constraints_met=7),
                                            dict(GOOD_ROWS[2], constraints_met=0, verdict="qualified")])
        result = self.run_script(STATUS, "--judge-dir", str(self.judge), "--json", "arm_a")
        self.assertEqual(result.returncode, 0, result.stderr)
        text = json.dumps(json.loads(result.stdout), ensure_ascii=False)
        self.assertIn("poiid 'PX' != input 'P1'", text)
        self.assertIn("constraints_met 7 outside 0..2", text)
        self.assertIn("qualified with constraints_met 0 != 1", text)
        self.assertEqual(json.loads(result.stdout)["labels"]["arm_a"]["totals"]["complete"], False)

    def test_json_output_is_machine_readable(self):
        write_fixture(self.judge, "arm_a", GOOD_ROWS)
        result = self.run_script(STATUS, "--judge-dir", str(self.judge), "--json", "arm_a")
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertTrue(payload["labels"]["arm_a"]["totals"]["complete"])
        self.assertEqual(payload["labels"]["arm_a"]["totals"]["expected_rows"], 3)

    def test_unknown_label_fails_clearly(self):
        write_fixture(self.judge, "arm_a", GOOD_ROWS)
        result = self.run_script(STATUS, "--judge-dir", str(self.judge), "arm_z")
        self.assertEqual(result.returncode, 1)
        self.assertIn("build_qoder_judge_inputs.py", result.stderr)


class TestEndToEndChain(SkillScriptCase):
    """Builder -> blind inputs -> verdicts -> published metrics, on a synthetic run."""

    TASKS = [
        {"id": "task-full", "query": "找适合阅读的咖啡店", "context": {"time": "2026-01-01 12:00:00"},
         "constraints": [{"cid": "c1", "text": "人均不超过50元", "h": "刚性", "private_annotation": "DO_NOT_EXPOSE"},
                         {"cid": "c2", "text": "适合安静阅读", "h": "柔性"}],
         "answer": {"satisfiable": True, "reference": "DO_NOT_EXPOSE"}, "target_poi": "DO_NOT_EXPOSE"},
        {"id": "task-empty", "query": "找不存在的店", "context": {}, "constraints": [],
         "answer": {"satisfiable": False}},
    ]
    RECORDS = [
        {"task_id": "task-full", "status": "completed", "input": {"query": "找适合阅读的咖啡店"},
         "model": "DO_NOT_EXPOSE",
         "answer": {"summary": "三家", "no_exact_match": False, "notes": [], "recommendations": [
             {"poiid": "p1", "name": "第一咖啡店", "reason": "人均40元", "evidence": ["poi:p1"], "constraint_gaps": []},
             {"poiid": "p2", "name": "第二咖啡店", "reason": "人均45元", "evidence": ["poi:p2"], "constraint_gaps": []},
             {"poiid": "p3", "name": "第三咖啡店", "reason": "人均60元", "evidence": ["poi:p3"], "constraint_gaps": ["超预算"]}]},
         "trajectory": [{"event": "tool_calls", "round": 1, "calls": [
             {"name": "poi.search", "arguments": {"query": "咖啡"},
              "observation": {"data": {"results": [
                  {"poiid": "p1", "name": "第一咖啡店", "evidence_id": "poi:p1", "average_cost": 40},
                  {"poiid": "p2", "name": "第二咖啡店", "evidence_id": "poi:p2", "average_cost": 45},
                  {"poiid": "p3", "name": "第三咖啡店", "evidence_id": "poi:p3", "average_cost": 60}]}}}]}],
         "metrics": {"tool_calls": 1, "elapsed_seconds": 0.5}},
        {"task_id": "task-empty", "status": "completed", "input": {"query": "找不存在的店"},
         "answer": {"summary": "无匹配", "no_exact_match": True, "notes": [], "recommendations": []},
         "trajectory": [], "metrics": {}},
    ]
    VERDICTS = [
        {"task_id": "task-full", "rank": 1, "poiid": "p1", "verdict": "qualified", "constraints_met": 2, "reason": "价格与安静均有自身证据"},
        {"task_id": "task-full", "rank": 2, "poiid": "p2", "verdict": "unacceptable", "constraints_met": 1, "reason": "安静无证据且未披露"},
        {"task_id": "task-full", "rank": 3, "poiid": "p3", "verdict": "acceptable", "constraints_met": 0, "reason": "超预算已披露"},
    ]

    def build_inputs(self, label):
        tasks = self.tmp / "tasks.json"
        tasks.write_text(json.dumps([{"record": t} for t in self.TASKS], ensure_ascii=False), encoding="utf-8")
        run = self.tmp / "run.jsonl"
        run.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in self.RECORDS), encoding="utf-8")
        result = self.run_script(PACKAGE / "judge/build_qoder_judge_inputs.py",
                                 "--run", run, "--label", label, "--judge-dir", self.judge, "--tasks", tasks)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_blind_inputs_then_published_metrics(self):
        self.build_inputs("arm_e2e")
        chunk = self.judge / "arm_e2e" / "inputs" / "chunk_01.json"
        self.assertTrue(chunk.is_file())
        raw = chunk.read_text(encoding="utf-8")
        self.assertNotIn("DO_NOT_EXPOSE", raw)
        payloads = json.loads(raw)
        self.assertEqual([p["task_id"] for p in payloads], ["task-full", "task-empty"])
        self.assertEqual(payloads[0]["n_constraints"], 2)
        self.assertIs(payloads[1]["satisfiable"], False)
        self.assertEqual([r["rank"] for r in payloads[0]["recommendations"]], [1, 2, 3])

        verdicts = self.judge / "arm_e2e" / "verdicts" / "chunk_01.jsonl"
        verdicts.write_text("".join(json.dumps(v, ensure_ascii=False) + "\n" for v in self.VERDICTS), encoding="utf-8")

        status = self.run_script(STATUS, "--judge-dir", str(self.judge), "arm_e2e")
        self.assertEqual(status.returncode, 0, status.stderr)
        self.assertIn("COMPLETE  chunks 1/1  queries 2  rows 3/3", status.stdout)

        result = self.run_script(SCORER, "--judge-dir", str(self.judge), "arm_e2e")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        summary = json.loads((self.judge / "arm_e2e" / "summary_published.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["queries"], 2)
        self.assertEqual(summary["should_answer_queries"], 1)
        self.assertEqual(summary["unsatisfiable_queries"], 1)
        self.assertEqual(summary["constraint_denominator"], 6)
        self.assertEqual(summary["csr"], 33.3333)
        self.assertEqual(summary["cr"], 50.0)
        self.assertEqual(summary["success_at_k"], 100.0)
        self.assertEqual(summary["recs_per_query"], 3.0)
        self.assertEqual(summary["coverage"], 1.0)


class TestRelocation(SkillScriptCase):
    """Both scripts must default to <relocated root>/judge without any flag."""

    def test_relocated_skill_defaults_to_its_own_judge_dir(self):
        root = self.tmp / "relocated bench"
        shutil.copytree(SKILL, root / ".qoder/skills/atlas-qoder-audit")
        write_fixture(root / "judge", "arm_a", GOOD_ROWS)
        caller = self.tmp / "elsewhere"
        caller.mkdir()
        status = self.run_script(root / ".qoder/skills/atlas-qoder-audit/scripts/audit_status.py",
                                 "arm_a", cwd=caller)
        self.assertEqual(status.returncode, 0, status.stderr)
        self.assertIn("COMPLETE  chunks 2/2", status.stdout)
        score = self.run_script(root / ".qoder/skills/atlas-qoder-audit/scripts/score_verdicts.py",
                                "--no-write", "arm_a", cwd=caller)
        self.assertEqual(score.returncode, 0, score.stderr)
        self.assertIn("| arm_a | 3/3 | 2 | 3/3 | 1.000 | 40.00 | 60.00 | 100.00 | 1.50 |", score.stdout)

    def test_relocated_skill_reports_a_missing_judge_dir(self):
        root = self.tmp / "relocated bench"
        shutil.copytree(SKILL, root / ".qoder/skills/atlas-qoder-audit")
        result = self.run_script(root / ".qoder/skills/atlas-qoder-audit/scripts/score_verdicts.py", cwd=root)
        self.assertEqual(result.returncode, 1)
        self.assertIn("No judge labels with inputs", result.stderr)


class TestSkillPackaging(unittest.TestCase):
    def test_scripts_are_executable_and_importable(self):
        for script in (STATUS, SCORER):
            self.assertTrue(script.is_file(), script)
            self.assertTrue(script.stat().st_mode & 0o111, f"{script} is not executable")
            compile(script.read_text(encoding="utf-8"), str(script), "exec")

    def test_default_judge_dir_is_the_repository_judge(self):
        for script in (STATUS, SCORER):
            name = f"_atlas_audit_{script.stem}"
            spec = importlib.util.spec_from_file_location(name, script)
            module = importlib.util.module_from_spec(spec)
            with mock.patch.object(sys, "dont_write_bytecode", True):
                spec.loader.exec_module(module)
            try:
                self.assertEqual(module.DEFAULT_JUDGE_DIR, PACKAGE / "judge", script)
                self.assertEqual(module.ROOT, PACKAGE, script)
            finally:
                sys.modules.pop(name, None)

    def test_frontmatter_matches_directory_and_limits(self):
        text = (SKILL / "SKILL.md").read_text(encoding="utf-8")
        self.assertTrue(text.startswith("---\n"))
        front, body = text[4:].split("\n---\n", 1)
        fields = dict(line.split(": ", 1) for line in front.splitlines() if ": " in line)
        self.assertEqual(fields["name"], SKILL.name)
        self.assertEqual(len(fields["description"].splitlines()), 1)
        self.assertLessEqual(len(fields["description"]), 1024)
        self.assertLess(len(body.splitlines()), 500)
        self.assertNotIn("TODO:", text)

    def test_skill_points_at_the_repository_rubric_and_builder(self):
        text = (SKILL / "SKILL.md").read_text(encoding="utf-8")
        for needle in ("judge/RUBRIC.md", "judge/build_qoder_judge_inputs.py",
                       "references/verdict-contract.md", "score_verdicts.py", "audit_status.py"):
            self.assertIn(needle, text)
        self.assertTrue((PACKAGE / "judge" / "RUBRIC.md").is_file())
        self.assertTrue((PACKAGE / "judge" / "build_qoder_judge_inputs.py").is_file())

    def test_main_table_scoring_scripts_live_beside_the_rubric(self):
        expected = [
            "build_qoder_judge_inputs.py", "paired_acceptability_judge.py", "factual_accuracy_judge.py",
            "combine_passrate_factual.py", "intent_understanding_judge.py", "tool_selection_judge.py",
            "evidence_sufficiency_judge.py", "tool_flexibility_judge.py", "summarize_intent_understanding.py",
            "summarize_tool_selection.py", "summarize_evidence_sufficiency.py", "summarize_tool_flexibility.py",
            "summarize_qoder_judge.py", "analyze_regional_results.py",
        ]
        for name in expected:
            self.assertTrue((PACKAGE / "judge" / name).is_file(), f"judge/{name} is missing")
            self.assertFalse((PACKAGE / "scripts" / name).exists(), f"scripts/{name} should have moved")
        for name in ("finalize_run.py", "run_agent.py"):
            self.assertTrue((PACKAGE / "scripts" / name).is_file(), f"scripts/{name} must stay")

    def test_contract_reference_documents_the_published_formulas(self):
        text = (SKILL / "references" / "verdict-contract.md").read_text(encoding="utf-8")
        for needle in ("GOOD = {qualified, acceptable}", "den", "Success@k", "Recs/query",
                       "summarize_qoder_judge.py", "1004"):
            self.assertIn(needle, text)


if __name__ == "__main__":
    unittest.main()
