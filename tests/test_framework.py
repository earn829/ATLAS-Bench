from __future__ import annotations

import json
import http.client
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from local_life_agent.agent import ReActAgent
from local_life_agent.config import AgentConfig, LLMConfig, PathConfig
from local_life_agent.dataset import AgentTask, load_tasks
from local_life_agent.dense_retrieval import VectorIndexError, resolve_model_source
from local_life_agent.index_builder import _district, build_indexes
from local_life_agent.neural_retrieval import VectorRerankPoiRepository
from local_life_agent.prompting import build_system_prompt
from local_life_agent.llm import (
    OpenAICompatibleClient,
    OutputBudgetExhausted,
    ScriptedLLMClient,
)
from local_life_agent.protocol import (
    FinalAction,
    ProtocolError,
    ToolBatchAction,
    parse_agent_message,
)
from local_life_agent.retrieval import PoiRepository
from local_life_agent.runner import run_batch
from local_life_agent.tools import build_default_registry


LEXICAL_TEST_CONFIG = AgentConfig(
    enable_vector_recall=False,
    vector_required=False,
)


def _poi(
    poiid: str,
    name: str,
    lng: float,
    lat: float,
    *,
    comments=None,
    category: str = "咖啡厅",
    rating: float = 4.6,
    average_cost: float = 38,
    star: float = 0,
    parking_spaces: int = 0,
    business_area: str = "望京",
):
    comments = comments or []
    return {
        "poiid": poiid,
        "name": name,
        "comment": int(bool(comments)),
        "poi_feature": {
            "basic_info": {
                "poiid": poiid,
                "name": name,
                "longitude": lng,
                "latitude": lat,
                "address": "北京市朝阳区测试路",
                "pcdr": "北京市;北京市;朝阳区",
                "tag_display_std": category,
            },
            "service_info": {
                "t_tag_class": "美食;咖啡厅",
                "business_area": business_area,
                "rating": rating,
                "averagecost": average_cost,
                "opentime_disp": "09:00-22:00",
                "atag": "有座位 WiFi",
                "star": star,
                "num_space_f": parking_spaces,
            },
        },
        "comment_features": comments,
    }


def _note(nid: str, content: str):
    return {
        "basic_info": {
            "nid": nid,
            "content": "笔记元信息\n笔记正文：" + content,
            "related_items": None,
            "loc": "116.46 40.0",
            "gmt_create": "2025-08-01 10:00:00",
            "gmt_modified": None,
            "t_tag2": "咖啡厅",
        },
        "details": {"open_tag": "安静学习", "hot_score": 9.0},
    }


class FrameworkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp_dir.name)
        cls.source = cls.root / "source.json"
        records = [
            _poi(
                "p1",
                "安静角落咖啡",
                116.4600,
                40.0000,
                comments=[
                    _note(
                        "n1",
                        "环境安静，插座很多，适合学习。" + "安静" * 700,
                    )
                ],
                star=5,
                parking_spaces=100,
            ),
            _poi(
                "p2",
                "晴天咖啡",
                116.4650,
                40.0000,
                rating=4.2,
                average_cost=68,
                star=4,
                parking_spaces=10,
            ),
            _poi(
                "p3",
                "远方咖啡",
                116.5500,
                40.0000,
                rating=4.9,
                average_cost=50,
                business_area="管庄",
            ),
            _poi(
                "p4",
                "无ID笔记咖啡",
                116.5600,
                40.0000,
                comments=[_note("", "有正文但没有可关联的nid")],
            ),
            _poi(
                "p5",
                "重复笔记咖啡",
                116.5700,
                40.0000,
                comments=[_note("n1", "重复nid不应制造错误comment标志")],
            ),
        ]
        cls.source.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
        cls.index_dir = cls.root / "indexes"
        cls.manifest = build_indexes(
            cls.source, cls.index_dir, progress_every=0
        )
        cls.prompt = cls.root / "prompt.md"
        cls.prompt.write_text("测试系统提示词", encoding="utf-8")
        cls.paths = PathConfig(
            poi_db=cls.index_dir / "poi_rag.sqlite3",
            note_db=cls.index_dir / "note_rag.sqlite3",
            prompt_file=cls.prompt,
            output_dir=cls.root / "outputs",
        )
        cls.registry = build_default_registry(cls.paths, LEXICAL_TEST_CONFIG)

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_index_split_and_four_tools(self):
        self.assertEqual(self.manifest["poi_count"], 5)
        self.assertEqual(self.manifest["note_count"], 1)
        self.assertEqual(self.manifest["missing_note_id_records"], 1)
        self.assertEqual(self.manifest["duplicate_note_records"], 1)
        self.assertEqual(
            self.registry.names(),
            ["search_pois", "search_notes", "get_poi_notes", "search_nearby"],
        )

        pois = self.registry.execute("search_pois", {"query": "咖啡", "top_k": 10})
        self.assertTrue(pois["ok"])
        by_id = {item["poiid"]: item for item in pois["data"]["results"]}
        self.assertEqual(by_id["p1"]["comment"], 1)
        self.assertEqual(by_id["p1"]["comment_count"], 1)
        self.assertEqual(by_id["p2"]["comment"], 0)
        self.assertEqual(by_id["p4"]["comment"], 0)
        self.assertEqual(by_id["p5"]["comment"], 0)

        notes = self.registry.execute("search_notes", {"query": "安静学习"})
        self.assertTrue(notes["ok"])
        self.assertEqual(notes["data"]["results"][0]["nid"], "n1")
        self.assertNotIn("笔记元信息", notes["data"]["results"][0]["content"])
        self.assertTrue(notes["data"]["results"][0]["content_truncated"])

        exact = self.registry.execute("get_poi_notes", {"poiid": "p1"})
        self.assertEqual(exact["data"]["count"], 1)
        self.assertEqual(exact["data"]["total_available"], 1)
        self.assertEqual(exact["data"]["notes"][0]["poiid"], "p1")
        duplicate = self.registry.execute("get_poi_notes", {"poiid": "p5"})
        self.assertEqual(duplicate["data"]["total_available"], 0)

        nearby = self.registry.execute(
            "search_nearby",
            {
                "query": "咖啡",
                "loc": {"lng": 116.46, "lat": 40.0},
                "distance_m": 1000,
                "top_k": 10,
            },
        )
        self.assertTrue(nearby["ok"])
        self.assertEqual(
            {item["poiid"] for item in nearby["data"]["results"]}, {"p1", "p2"}
        )
        self.assertTrue(
            all(item["distance_m"] <= 1000 for item in nearby["data"]["results"])
        )

    def test_district_parser_prefers_specific_district_over_autonomous_region(self):
        self.assertEqual(
            _district("新疆维吾尔自治区;乌鲁木齐市;达坂城区;"),
            "达坂城区",
        )
        self.assertEqual(_district("北京市;;朝阳区;望京街"), "朝阳区")
        self.assertEqual(
            _district("北京市;;朝阳区;和平街十三区"), "朝阳区"
        )

    def test_multiple_sources_build_one_unified_index(self):
        with tempfile.TemporaryDirectory() as temp_value:
            root = Path(temp_value)
            first = root / "first.json"
            second = root / "second.json"
            first.write_text(
                json.dumps(
                    [_poi("region-a", "甲地区咖啡", 116.4, 40.0)],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            second.write_text(
                json.dumps(
                    [
                        _poi(
                            "region-b",
                            "乙地区咖啡",
                            121.4,
                            31.2,
                            comments=[_note("region-note", "乙地区安静咖啡")],
                        )
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            index_dir = root / "indexes"
            manifest = build_indexes(
                [first, second], index_dir, progress_every=0
            )
            self.assertEqual(manifest["poi_count"], 2)
            self.assertEqual(manifest["note_count"], 1)
            self.assertEqual(manifest["source_records_read"], 2)
            self.assertEqual(len(manifest["sources"]), 2)
            paths = PathConfig(
                poi_db=index_dir / "poi_rag.sqlite3",
                note_db=index_dir / "note_rag.sqlite3",
                prompt_file=self.prompt,
                output_dir=root / "outputs",
            )
            registry = build_default_registry(paths, LEXICAL_TEST_CONFIG)
            for poiid in ("region-a", "region-b"):
                result = registry.execute("search_pois", {"query": poiid})
                self.assertEqual(result["data"]["results"][0]["poiid"], poiid)

    def test_chinese_recall_exact_poiid_and_structured_filters(self):
        for query in ("望京 咖啡", "想喝咖啡", "安静 咖啡店"):
            result = self.registry.execute("search_pois", {"query": query, "top_k": 10})
            self.assertTrue(result["ok"], query)
            self.assertTrue(result["data"]["results"], query)
        ranked = self.registry.execute(
            "search_pois", {"query": "望京 咖啡", "top_k": 10}
        )
        self.assertIn("望京", ranked["data"]["results"][0]["business_area"])

        exact = self.registry.execute("search_pois", {"query": "p1", "top_k": 10})
        self.assertEqual([item["poiid"] for item in exact["data"]["results"]], ["p1"])

        filtered = self.registry.execute(
            "search_pois",
            {
                "query": "咖啡",
                "min_rating": 4.5,
                "max_average_cost": 40,
                "has_notes": True,
                "min_star": 5,
                "min_parking_spaces": 50,
                "top_k": 10,
            },
        )
        self.assertTrue(filtered["ok"])
        self.assertEqual(
            [item["poiid"] for item in filtered["data"]["results"]], ["p1"]
        )

    def test_nearby_rtree_boundary_and_strict_inputs(self):
        exact_origin = self.registry.execute(
            "search_nearby",
            {
                "query": "咖啡",
                "loc": {"lng": 116.465, "lat": 40.0},
                "distance_m": 1,
                "top_k": 5,
            },
        )
        self.assertTrue(exact_origin["ok"])
        self.assertIn("p2", {item["poiid"] for item in exact_origin["data"]["results"]})

        nan_result = self.registry.execute(
            "search_nearby",
            {
                "query": "咖啡",
                "loc": {"lng": float("nan"), "lat": 40.0},
                "distance_m": 1000,
            },
        )
        self.assertFalse(nan_result["ok"])
        bad_top_k = self.registry.execute(
            "search_pois", {"query": "咖啡", "top_k": 1.9}
        )
        self.assertFalse(bad_top_k["ok"])
        unknown = self.registry.execute(
            "search_pois", {"query": "咖啡", "unexpected": True}
        )
        self.assertFalse(unknown["ok"])

    def test_max_top_k_propagates_to_tool_schemas_and_prompt(self):
        cap = LEXICAL_TEST_CONFIG.max_top_k
        at_cap = self.registry.execute("search_pois", {"query": "咖啡", "top_k": cap})
        self.assertTrue(at_cap["ok"])
        over_cap = self.registry.execute(
            "search_pois", {"query": "咖啡", "top_k": cap + 1}
        )
        self.assertFalse(over_cap["ok"])

        advertised = [
            spec["input_schema"]["properties"]["top_k"]["maximum"]
            for spec in self.registry.specifications()
            if "top_k" in spec["input_schema"]["properties"]
        ]
        self.assertEqual(len(advertised), 3)
        self.assertEqual(set(advertised), {cap})

        prompt = build_system_prompt(
            self.registry, paths=self.paths, config=LEXICAL_TEST_CONFIG
        )
        self.assertIn('"max_top_k": %d' % cap, prompt)

    def test_failed_overwrite_preserves_published_indexes(self):
        malformed = self.root / "malformed.json"
        malformed.write_text('[{"poiid":"broken"', encoding="utf-8")
        with self.assertRaises(ValueError):
            build_indexes(
                malformed,
                self.index_dir,
                overwrite=True,
                progress_every=0,
            )
        registry = build_default_registry(self.paths, LEXICAL_TEST_CONFIG)
        result = registry.execute("search_pois", {"query": "咖啡"})
        self.assertTrue(result["ok"])
        self.assertTrue(result["data"]["results"])

    def test_build_version_guard_rejects_mixed_publish(self):
        with tempfile.TemporaryDirectory() as temp_value:
            root = Path(temp_value)
            source = root / "source.json"
            source.write_text(self.source.read_text(encoding="utf-8"), encoding="utf-8")
            index_dir = root / "indexes"
            build_indexes(source, index_dir, progress_every=0)
            paths = PathConfig(
                poi_db=index_dir / "poi_rag.sqlite3",
                note_db=index_dir / "note_rag.sqlite3",
                prompt_file=self.prompt,
                output_dir=root / "outputs",
            )
            self.assertTrue(
                build_default_registry(paths, LEXICAL_TEST_CONFIG).names()
            )

            real_replace = __import__("os").replace

            def fail_note_publish(source_path, destination_path):
                if str(destination_path).endswith("note_rag.sqlite3"):
                    raise OSError("模拟第二个数据库发布失败")
                return real_replace(source_path, destination_path)

            with mock.patch(
                "local_life_agent.index_builder.os.replace",
                side_effect=fail_note_publish,
            ):
                with self.assertRaises(OSError):
                    build_indexes(
                        source, index_dir, overwrite=True, progress_every=0
                    )
            with self.assertRaises(RuntimeError):
                build_default_registry(paths, LEXICAL_TEST_CONFIG)

    def test_note_search_diversifies_pois(self):
        with tempfile.TemporaryDirectory() as temp_value:
            root = Path(temp_value)
            source = root / "source.json"
            records = [
                _poi(
                    "d1",
                    "甲咖啡",
                    116.4,
                    40.0,
                    comments=[
                        _note("dn1", "环境安静适合学习"),
                        _note("dn2", "环境安静有插座"),
                        _note("dn3", "环境安静座位多"),
                        _note("dn4", "环境安静服务好"),
                    ],
                ),
                _poi(
                    "d2",
                    "乙咖啡",
                    116.41,
                    40.0,
                    comments=[_note("dn5", "环境安静适合聊天")],
                ),
            ]
            source.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
            index_dir = root / "indexes"
            build_indexes(source, index_dir, progress_every=0)
            registry = build_default_registry(
                PathConfig(
                    poi_db=index_dir / "poi_rag.sqlite3",
                    note_db=index_dir / "note_rag.sqlite3",
                    prompt_file=self.prompt,
                    output_dir=root / "outputs",
                ),
                LEXICAL_TEST_CONFIG,
            )
            result = registry.execute(
                "search_notes", {"query": "环境安静", "top_k": 3}
            )
            poiids = [item["poiid"] for item in result["data"]["results"]]
            self.assertEqual(len(poiids), 3)
            self.assertIn("d2", poiids)
            self.assertLessEqual(poiids.count("d1"), 2)

    def test_qwen_vector_candidates_are_reranked_without_lexical_fusion(self):
        class FakeVectorIndex:
            def search_ids(self, query, top_k, *, eligible_ids=None):
                self.last_query = query
                self.last_top_k = top_k
                return [(3, 0.91), (1, 0.72)]

        class FakeReranker:
            def score(self, query, documents):
                self.last_query = query
                self.last_documents = documents
                return [0.1 if "远方咖啡" in document else 0.9 for document in documents]

        vector = FakeVectorIndex()
        reranker = FakeReranker()
        repository = VectorRerankPoiRepository(
            PoiRepository(
                self.paths.poi_db,
                expected_build_id=self.manifest["build_id"],
            ),
            vector,
            reranker,
            candidate_k=100,
        )
        results = repository.search("安静", top_k=2)
        self.assertEqual([item["poiid"] for item in results], ["p1", "p3"])
        self.assertEqual(results[0]["vector_rank"], 2)
        self.assertEqual(results[0]["rerank_rank"], 1)
        self.assertEqual(results[0]["vector_similarity"], 0.72)
        self.assertEqual(results[0]["rerank_score"], 0.9)
        self.assertEqual(vector.last_query, "安静")
        self.assertEqual(vector.last_top_k, 100)
        self.assertEqual(reranker.last_query, "安静")

    def test_qwen_vector_search_applies_structured_filter_before_candidate_cutoff(self):
        class FakeVectorIndex:
            def search_ids(self, query, top_k, *, eligible_ids=None):
                self.eligible_ids = list(eligible_ids or [])
                ranking = [(1, 0.99), (3, 0.75)]
                allowed = set(self.eligible_ids)
                return [pair for pair in ranking if pair[0] in allowed][:top_k]

        class FakeReranker:
            def score(self, query, documents):
                return [0.8 for _ in documents]

        vector = FakeVectorIndex()
        repository = VectorRerankPoiRepository(
            PoiRepository(
                self.paths.poi_db,
                expected_build_id=self.manifest["build_id"],
            ),
            vector,
            FakeReranker(),
            candidate_k=100,
        )
        results = repository.search("咖啡", top_k=10, filters={"min_rating": 4.8})
        self.assertEqual([item["poiid"] for item in results], ["p3"])
        self.assertIn(3, vector.eligible_ids)
        self.assertNotIn(1, vector.eligible_ids)

    def test_agent_react_and_multi_poi_grounding(self):
        responses = [
            json.dumps(
                {
                    "type": "tool_calls",
                    "calls": [
                        {
                            "call_id": "c1",
                            "tool": "search_pois",
                            "arguments": {"query": "咖啡", "top_k": 10},
                            "rationale": "找咖啡候选",
                        },
                        {
                            "call_id": "c2",
                            "tool": "search_notes",
                            "arguments": {"query": "环境安静", "top_k": 10},
                            "rationale": "找安静体验证据",
                        },
                    ],
                },
                ensure_ascii=False,
            ),
            json.dumps(
                {
                    "type": "final",
                    "answer": {
                        "summary": "两个候选",
                        "recommendations": [
                            {
                                "rank": 1,
                                "poiid": "p1",
                                "name": "安静角落咖啡",
                                "reason": "笔记明确提到环境安静且有插座。",
                                "evidence": ["poi:p1", "note:n1"],
                                "constraint_gaps": [],
                            },
                            {
                                "rank": 2,
                                "poiid": "p2",
                                "name": "晴天咖啡",
                                "reason": "POI结构信息显示是咖啡厅且位于望京。",
                                "evidence": ["poi:p2"],
                                "constraint_gaps": ["本地无笔记，无法核验安静程度"],
                            },
                        ],
                        "no_exact_match": False,
                        "notes": "",
                    },
                },
                ensure_ascii=False,
            ),
        ]
        llm = ScriptedLLMClient(responses)
        agent = ReActAgent(
            llm=llm,
            tools=self.registry,
            config=AgentConfig(),
            paths=self.paths,
        )
        result = agent.run(
            {
                "query": "望京适合学习的咖啡店",
                "time": "2025-08-02 14:00:00",
                "loc": {"lng": 116.46, "lat": 40.0},
            }
        )
        self.assertEqual(result["status"], "completed", result)
        self.assertEqual(len(result["answer"]["recommendations"]), 2)
        self.assertEqual(result["metrics"]["tool_rounds"], 1)
        self.assertEqual(result["metrics"]["tool_calls"], 2)
        self.assertEqual(result["trajectory"][0]["event"], "tool_calls")
        self.assertTrue(result["trajectory"][0]["parallel"])
        self.assertEqual(
            [call["call_id"] for call in result["trajectory"][0]["calls"]],
            ["c1", "c2"],
        )
        self.assertIn("TOOL_OBSERVATIONS", llm.calls[1][-1]["content"])
        first_user_message = llm.calls[0][1]["content"]
        self.assertNotIn("task_id", first_user_message)
        self.assertNotIn("id\"", first_user_message)

    def test_hallucinated_poi_is_rejected_then_repaired(self):
        bad = {
            "type": "final",
            "answer": {
                "summary": "",
                "recommendations": [
                    {
                        "rank": 1,
                        "poiid": "invented",
                        "name": "不存在",
                        "reason": "猜的",
                        "evidence": ["poi:invented"],
                        "constraint_gaps": [],
                    }
                ],
                "no_exact_match": False,
            },
        }
        search = {
            "type": "tool_calls",
            "calls": [
                {
                    "call_id": "c1",
                    "tool": "search_pois",
                    "arguments": {"query": "咖啡", "top_k": 2},
                    "rationale": "先获取真实候选",
                }
            ],
        }
        repaired = {
            "type": "final",
            "answer": {
                "summary": "已改为工具核验候选",
                "recommendations": [
                    {
                        "rank": 1,
                        "poiid": "p2",
                        "name": "晴天咖啡",
                        "reason": "POI工具确认该地点属于咖啡厅。",
                        "evidence": ["poi:p2"],
                        "constraint_gaps": [],
                    }
                ],
                "no_exact_match": False,
                "notes": "",
            },
        }
        llm = ScriptedLLMClient(
            [
                json.dumps(bad, ensure_ascii=False),
                json.dumps(search, ensure_ascii=False),
                json.dumps(repaired, ensure_ascii=False),
            ]
        )
        agent = ReActAgent(
            llm=llm,
            tools=self.registry,
            config=AgentConfig(
                max_tool_rounds=1,
                max_tools_per_round=2,
                max_tool_calls=2,
                max_parse_retries=1,
            ),
            paths=self.paths,
        )
        result = agent.run(
            {
                "query": "找咖啡",
                "time": "2025-08-02 14:00:00",
                "loc": {"lng": 116.46, "lat": 40.0},
            }
        )
        self.assertEqual(result["status"], "completed", result)
        self.assertEqual(result["answer"]["recommendations"][0]["poiid"], "p2")
        self.assertEqual(result["trajectory"][0]["event"], "protocol_error")

    def test_final_semantic_violations_are_judged_without_retry(self):
        search = {
            "type": "tool_calls",
            "calls": [
                {
                    "call_id": "c1",
                    "tool": "search_pois",
                    "arguments": {"query": "咖啡", "top_k": 2},
                }
            ],
        }
        inconsistent = {
            "type": "final",
            "answer": {
                "summary": "保留模型第一次最终答案",
                "recommendations": [
                    {
                        "rank": 1,
                        "poiid": "p2",
                        "name": "晴天咖啡",
                        "reason": "错误引用了另一家店的证据",
                        "evidence": ["poi:p3"],
                        "constraint_gaps": [],
                    }
                ],
                "no_exact_match": True,
            },
        }
        llm = ScriptedLLMClient(
            [
                json.dumps(search, ensure_ascii=False),
                json.dumps(inconsistent, ensure_ascii=False),
            ]
        )
        result = ReActAgent(
            llm=llm,
            tools=self.registry,
            config=AgentConfig(),
            paths=self.paths,
        ).run(
            {
                "query": "找咖啡",
                "time": "2025-08-02 14:00:00",
                "loc": {"lng": 116.46, "lat": 40.0},
            }
        )
        self.assertEqual(result["status"], "completed", result)
        self.assertEqual(result["metrics"]["model_calls"], 2)
        self.assertFalse(
            any(event["event"] == "protocol_error" for event in result["trajectory"])
        )
        judge = result["programmatic_judge"]
        self.assertFalse(judge["passed"])
        self.assertFalse(judge["no_exact_match"]["consistent"])
        self.assertEqual(judge["evidence_ownership"]["reference_count"], 1)
        self.assertEqual(judge["evidence_ownership"]["wrong_owner_count"], 1)
        self.assertEqual(judge["evidence_ownership"]["ownership_accuracy"], 0.0)

    def test_batch_summary_aggregates_programmatic_judge_metrics(self):
        class FakeAgent:
            def run(self, _agent_input):
                return {
                    "status": "completed",
                    "answer": {},
                    "programmatic_judge": {
                        "passed": False,
                        "no_exact_match": {"consistent": False},
                        "evidence_ownership": {
                            "reference_count": 2,
                            "known_count": 1,
                            "correct_owner_count": 1,
                            "wrong_owner_count": 0,
                            "unknown_count": 1,
                        },
                    },
                    "trajectory": [],
                    "metrics": {},
                }

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "result.jsonl"
            summary = run_batch(
                [
                    AgentTask(
                        id="judge-case",
                        query="测试",
                        time="2025-08-02 14:00:00",
                        loc={"lng": 116.46, "lat": 40.0},
                    )
                ],
                agent_factory=FakeAgent,
                output_path=output,
                workers=1,
                resume=False,
            )
        judge = summary["programmatic_judge"]
        self.assertEqual(judge["scope"], "executed_records")
        self.assertEqual(judge["evaluated_answers"], 1)
        self.assertEqual(judge["no_exact_match_consistency_rate"], 0.0)
        self.assertEqual(judge["evidence_existence_accuracy"], 0.5)
        self.assertEqual(judge["evidence_ownership_accuracy"], 0.5)

    def test_model_source_prefers_published_local_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache_dir = Path(temporary)
            model_root = cache_dir / "models--Qwen--Example"
            snapshot = model_root / "snapshots" / "revision123"
            snapshot.mkdir(parents=True)
            (model_root / "refs").mkdir()
            (model_root / "refs" / "main").write_text(
                "revision123\n", encoding="utf-8"
            )
            self.assertEqual(
                resolve_model_source(
                    "Qwen/Example", cache_dir, local_files_only=True
                ),
                str(snapshot.resolve()),
            )

    def test_model_source_rejects_missing_offline_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(VectorIndexError):
                resolve_model_source(
                    "Qwen/Missing", Path(temporary), local_files_only=True
                )

    def test_protocol_repairs_unescaped_quotes_inside_string_values(self):
        raw = (
            '{"type":"final","answer":{"summary":"没有找到同时满足'
            '"开了很多年"和"出餐快"的门店","recommendations":[],'
            '"no_exact_match":true}}'
        )
        action = parse_agent_message(raw)
        self.assertIsInstance(action, FinalAction)
        self.assertEqual(
            action.answer["summary"],
            '没有找到同时满足"开了很多年"和"出餐快"的门店',
        )

    def test_protocol_does_not_repair_structural_json_errors(self):
        raw = (
            '{"type":"final" "answer":{"summary":"没有结果",'
            '"recommendations":[],"no_exact_match":true}}'
        )
        with self.assertRaises(ProtocolError):
            parse_agent_message(raw)

    def test_tool_batch_protocol_enforces_two_call_limit_and_unique_ids(self):
        valid = parse_agent_message(
            json.dumps(
                {
                    "type": "tool_calls",
                    "calls": [
                        {
                            "call_id": "c1",
                            "tool": "search_pois",
                            "arguments": {"query": "咖啡"},
                        },
                        {
                            "call_id": "c2",
                            "tool": "search_notes",
                            "arguments": {"query": "安静"},
                        },
                    ],
                },
                ensure_ascii=False,
            )
        )
        self.assertIsInstance(valid, ToolBatchAction)
        self.assertEqual(len(valid.calls), 2)

        three_calls = {
            "type": "tool_calls",
            "calls": [
                {"call_id": "c1", "tool": "a", "arguments": {}},
                {"call_id": "c2", "tool": "b", "arguments": {}},
                {"call_id": "c3", "tool": "c", "arguments": {}},
            ],
        }
        with self.assertRaises(ProtocolError):
            parse_agent_message(json.dumps(three_calls))

        duplicate_ids = {
            "type": "tool_calls",
            "calls": [
                {"call_id": "same", "tool": "a", "arguments": {}},
                {"call_id": "same", "tool": "b", "arguments": {}},
            ],
        }
        with self.assertRaises(ProtocolError):
            parse_agent_message(json.dumps(duplicate_ids))

    def test_agent_config_has_absolute_tool_limits(self):
        with self.assertRaises(ValueError):
            AgentConfig(max_tool_rounds=6)
        with self.assertRaises(ValueError):
            AgentConfig(max_tools_per_round=3)
        with self.assertRaises(ValueError):
            AgentConfig(max_tool_calls=11)

    def test_agent_stops_after_five_tool_rounds_and_ten_calls(self):
        batch = json.dumps(
            {
                "type": "tool_calls",
                "calls": [
                    {
                        "call_id": "poi",
                        "tool": "search_pois",
                        "arguments": {"query": "咖啡", "top_k": 1},
                    },
                    {
                        "call_id": "note",
                        "tool": "search_notes",
                        "arguments": {"query": "环境安静", "top_k": 1},
                    },
                ],
            },
            ensure_ascii=False,
        )
        final = json.dumps(
            {
                "type": "final",
                "answer": {
                    "summary": "已按预算完成检索",
                    "recommendations": [
                        {
                            "rank": 1,
                            "poiid": "p2",
                            "name": "晴天咖啡",
                            "reason": "POI工具确认该候选为咖啡厅。",
                            "evidence": ["poi:p2"],
                            "constraint_gaps": [],
                        }
                    ],
                    "no_exact_match": False,
                },
            },
            ensure_ascii=False,
        )
        llm = ScriptedLLMClient([batch] * 5 + [final])
        result = ReActAgent(
            llm=llm,
            tools=self.registry,
            config=AgentConfig(),
            paths=self.paths,
        ).run(
            {
                "query": "找咖啡",
                "time": "2025-08-02 14:00:00",
                "loc": {"lng": 116.46, "lat": 40.0},
            }
        )
        self.assertEqual(result["status"], "completed", result)
        self.assertEqual(result["metrics"]["tool_rounds"], 5)
        self.assertEqual(result["metrics"]["tool_calls"], 10)
        self.assertTrue(result["trajectory"][-1]["forced"])

    def test_dataset_whitelist(self):
        dataset = self.root / "tasks.json"
        dataset.write_text(
            json.dumps(
                {
                    "records": [
                        {
                            "id": "case-1",
                            "query": "附近咖啡",
                            "context": {
                                "time": "2025-08-01 09:00:00",
                                "loc": {"lng": 116.4, "lat": 39.9},
                                "secret": "不能给agent",
                            },
                            "reference_answer": "不能给agent",
                        }
                    ]
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        task = load_tasks(dataset)[0]
        self.assertEqual(set(task.to_agent_input()), {"query", "time", "loc"})
        self.assertNotIn("id", task.to_agent_input())

    def test_exact_llm_endpoint_is_not_rewritten(self):
        config = LLMConfig(
            base_url="https://example.invalid/v1",
            endpoint_url="https://example.invalid/open_api/v1/chat",
            api_key="test-only",
            model="test-model",
        )
        client = OpenAICompatibleClient(config)
        self.assertEqual(
            client._endpoint(), "https://example.invalid/open_api/v1/chat"
        )

    def test_chat_completions_can_disable_thinking(self):
        config = LLMConfig(
            base_url="https://example.invalid/v1",
            api_key="test-only",
            model="test-model",
            enable_thinking=False,
        )
        response = mock.MagicMock()
        response.read.return_value = (
            b'{"choices":[{"message":{"content":"ok"}}],"usage":{}}'
        )
        response.__enter__.return_value = response
        with mock.patch(
            "local_life_agent.llm.urllib.request.urlopen", return_value=response
        ) as urlopen:
            result = OpenAICompatibleClient(config).complete(
                [{"role": "user", "content": "hello"}]
            )
        request = urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(result["content"], "ok")
        self.assertIs(
            payload["extend_fields"]["chat_template_kwargs"]["enable_thinking"],
            False,
        )

    def test_thinking_field_can_target_vllm_chat_template_kwargs(self):
        config = LLMConfig(
            base_url="https://example.invalid/v1",
            api_key="test-only",
            model="test-model",
            enable_thinking=False,
            thinking_field="chat_template_kwargs",
        )
        response = mock.MagicMock()
        response.read.return_value = (
            b'{"choices":[{"message":{"content":"ok"}}],"usage":{}}'
        )
        response.__enter__.return_value = response
        with mock.patch(
            "local_life_agent.llm.urllib.request.urlopen", return_value=response
        ) as urlopen:
            OpenAICompatibleClient(config).complete(
                [{"role": "user", "content": "hello"}]
            )
        payload = json.loads(urlopen.call_args.args[0].data.decode("utf-8"))
        self.assertIs(payload["chat_template_kwargs"]["enable_thinking"], False)
        self.assertNotIn("extend_fields", payload)

    def test_chat_completions_can_cache_system_prompt(self):
        config = LLMConfig(
            base_url="https://example.invalid/v1",
            api_key="test-only",
            model="claude-test",
            cache_system_prompt=True,
        )
        response = mock.MagicMock()
        response.read.return_value = (
            b'{"choices":[{"message":{"content":"ok"}}],"usage":{}}'
        )
        response.__enter__.return_value = response
        messages = [
            {"role": "system", "content": "system rules"},
            {"role": "user", "content": "hello"},
        ]
        with mock.patch(
            "local_life_agent.llm.urllib.request.urlopen", return_value=response
        ) as urlopen:
            OpenAICompatibleClient(config).complete(messages)
        payload = json.loads(urlopen.call_args.args[0].data.decode("utf-8"))
        self.assertEqual(
            payload["messages"],
            [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "system rules",
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                },
                {"role": "user", "content": "hello"},
            ],
        )
        self.assertEqual(messages[0]["content"], "system rules")

    def test_invalid_thinking_field_is_rejected(self):
        with self.assertRaises(ValueError):
            LLMConfig(
                base_url="https://example.invalid/v1",
                api_key="test-only",
                model="test-model",
                thinking_field="reasoning",
            )

    def test_exhausted_output_budget_fails_fast_without_retry(self):
        config = LLMConfig(
            base_url="https://example.invalid/v1",
            api_key="test-only",
            model="test-model",
            max_retries=3,
        )
        response = mock.MagicMock()
        response.read.return_value = (
            b'{"choices":[{"message":{"content":null},"finish_reason":"length"}],'
            b'"usage":{"completion_tokens":4096}}'
        )
        response.__enter__.return_value = response
        with mock.patch(
            "local_life_agent.llm.urllib.request.urlopen", return_value=response
        ) as urlopen:
            with self.assertRaises(OutputBudgetExhausted):
                OpenAICompatibleClient(config).complete(
                    [{"role": "user", "content": "hello"}]
                )
        self.assertEqual(urlopen.call_count, 1)

    def test_remote_disconnect_is_retried_instead_of_failing_the_task(self):
        config = LLMConfig(
            base_url="https://example.invalid/v1",
            api_key="test-only",
            model="test-model",
            max_retries=3,
        )
        good = mock.MagicMock()
        good.read.return_value = b'{"choices":[{"message":{"content":"ok"}}],"usage":{}}'
        good.__enter__.return_value = good
        disconnect = http.client.RemoteDisconnected("Remote end closed connection without response")

        with mock.patch("local_life_agent.llm.time.sleep"):
            with mock.patch(
                "local_life_agent.llm.urllib.request.urlopen",
                side_effect=[disconnect, good],
            ) as urlopen:
                result = OpenAICompatibleClient(config).complete(
                    [{"role": "user", "content": "hello"}]
                )
            self.assertEqual(result["content"], "ok")
            self.assertEqual(urlopen.call_count, 2)

            with mock.patch(
                "local_life_agent.llm.urllib.request.urlopen", side_effect=disconnect
            ) as always_down:
                with self.assertRaises(RuntimeError):
                    OpenAICompatibleClient(config).complete(
                        [{"role": "user", "content": "hello"}]
                    )
            self.assertEqual(always_down.call_count, 3)

    def test_optional_generation_parameters_sent_only_when_configured(self):
        def payload_for(**kwargs):
            config = LLMConfig(
                base_url="https://example.invalid/v1",
                api_key="test-only",
                model="test-model",
                **kwargs,
            )
            response = mock.MagicMock()
            response.read.return_value = (
                b'{"choices":[{"message":{"content":"ok"}}],"usage":{}}'
            )
            response.__enter__.return_value = response
            with mock.patch(
                "local_life_agent.llm.urllib.request.urlopen", return_value=response
            ) as urlopen:
                OpenAICompatibleClient(config).complete(
                    [{"role": "user", "content": "hello"}]
                )
            return json.loads(urlopen.call_args.args[0].data.decode("utf-8"))

        self.assertEqual(payload_for(repetition_penalty=1.05)["repetition_penalty"], 1.05)
        self.assertNotIn("repetition_penalty", payload_for())
        self.assertEqual(payload_for(reasoning_effort="low")["reasoning_effort"], "low")
        self.assertNotIn("reasoning_effort", payload_for())
        with self.assertRaises(ValueError):
            LLMConfig(
                base_url="https://example.invalid/v1",
                api_key="test-only",
                model="test-model",
                repetition_penalty=2.5,
            )

    def test_finalize_run_dedupe_prefers_completed_and_keeps_runner_ownership_rule(self):
        import importlib.util

        script = Path(__file__).resolve().parents[1] / "scripts" / "finalize_run.py"
        spec = importlib.util.spec_from_file_location("finalize_run", script)
        finalize = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(finalize)

        completed_judge = {
            "passed": True,
            "no_exact_match": {"consistent": True},
            "evidence_ownership": {
                "reference_count": 2,
                "known_count": 1,
                "correct_owner_count": 1,
                "wrong_owner_count": 0,
                "unknown_count": 1,
            },
        }
        lines = [
            json.dumps(
                {
                    "task_id": "t1",
                    "status": "failed",
                    "error": {"type": "OutputBudgetExhausted"},
                }
            ),
            json.dumps(
                {
                    "task_id": "t2",
                    "status": "failed",
                    "error": {"type": "RemoteDisconnected", "message": "第一次"},
                }
            ),
            json.dumps(
                {
                    "task_id": "t1",
                    "status": "completed",
                    "answer": {"recommendations": [], "no_exact_match": True},
                    "programmatic_judge": completed_judge,
                    "metrics": {"tool_calls": 4, "elapsed_seconds": 12.5, "usage_events": []},
                }
            ),
            json.dumps(
                {
                    "task_id": "t2",
                    "status": "failed",
                    "error": {"type": "RemoteDisconnected", "message": "第二次"},
                }
            ),
        ]

        records, attempts, parse_errors = finalize.dedupe_records(lines)
        self.assertEqual(dict(parse_errors), {})
        self.assertEqual(dict(attempts), {"t1": 2, "t2": 2})
        by_id = {record["task_id"]: record for record in records}
        self.assertEqual(len(records), 2)
        self.assertEqual(by_id["t1"]["status"], "completed")
        self.assertEqual(by_id["t2"]["error"]["message"], "第二次")

        summary = finalize.summarize(
            records, raw_line_count=len(lines), attempts=attempts, config={}
        )
        self.assertEqual(summary["completed"], 1)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["tasks_with_multiple_attempts"], 2)
        self.assertEqual(summary["failure_types"], {"RemoteDisconnected": 1})
        judge = summary["programmatic_judge"]
        self.assertEqual(judge["answer_pass_rate"], 1.0)
        self.assertEqual(judge["evidence_existence_accuracy"], 0.5)
        self.assertEqual(judge["evidence_ownership_accuracy"], 0.5)

    def test_finalize_run_reads_unicode_line_separators_inside_json_strings(self):
        import importlib.util

        script = Path(__file__).resolve().parents[1] / "scripts" / "finalize_run.py"
        spec = importlib.util.spec_from_file_location("finalize_run", script)
        finalize = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(finalize)

        record = {
            "task_id": "unicode-separator",
            "status": "completed",
            "answer": {"summary": "第一行\u2028第二行\u2029第三行"},
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.jsonl"
            path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
            lines = finalize.read_jsonl(path)

        self.assertEqual(len(lines), 1)
        records, attempts, parse_errors = finalize.dedupe_records(lines)
        self.assertEqual(dict(parse_errors), {})
        self.assertEqual(dict(attempts), {"unicode-separator": 1})
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["answer"]["summary"], "第一行\u2028第二行\u2029第三行")

    def test_anthropic_messages_request_and_response(self):
        config = LLMConfig(
            base_url="https://example.invalid/protocol/anthropic/v1",
            api_key="test-only",
            model="claude-test",
            api_style="anthropic",
            omit_temperature=True,
        )
        response = mock.MagicMock()
        response.read.return_value = (
            b'{"id":"msg_test","content":[{"type":"text","text":"ok"}],'
            b'"usage":{"input_tokens":7,"output_tokens":1}}'
        )
        response.__enter__.return_value = response
        with mock.patch(
            "local_life_agent.llm.urllib.request.urlopen", return_value=response
        ) as urlopen:
            result = OpenAICompatibleClient(config).complete(
                [
                    {"role": "system", "content": "system rules"},
                    {"role": "user", "content": "hello"},
                ]
            )
        request = urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        headers = {key.lower(): value for key, value in request.header_items()}
        self.assertEqual(
            request.full_url,
            "https://example.invalid/protocol/anthropic/v1/messages",
        )
        self.assertEqual(headers["x-api-key"], "test-only")
        self.assertEqual(headers["anthropic-version"], "2023-06-01")
        self.assertNotIn("authorization", headers)
        self.assertEqual(payload["system"], "system rules")
        self.assertEqual(payload["messages"], [{"role": "user", "content": "hello"}])
        self.assertNotIn("temperature", payload)
        self.assertIn("\n\nuser TOOL_OBSERVATION", payload["stop_sequences"])
        self.assertIn("\n\nhuman", payload["stop_sequences"])
        self.assertEqual(result["content"], "ok")
        self.assertEqual(result["usage"], {"input_tokens": 7, "output_tokens": 1})


if __name__ == "__main__":
    unittest.main()
