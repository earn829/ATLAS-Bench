"""Blind paired acceptability audit of two agent runs.

Each task's recommendations from both arms are anonymized as system A/B (assignment
derived from a hash of the task id) and judged per recommendation against the human
constraint annotations, with no reference answer and no click target in the prompt.
Verdicts: qualified / acceptable / unacceptable.

The judge prompt and payload shape are fixed; changing them breaks comparability with
outputs/deepseek-v4-pro-thinking-1to10-100/ACCEPTABILITY_VS_QWEN36.md.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import sys
import threading
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Tuple

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "scripts"))
sys.path.insert(0, str(_HERE))

from finalize_run import dedupe_records, read_jsonl  # noqa: E402

VERDICTS = {"qualified", "acceptable", "unacceptable"}

SYSTEM = '''你是严格的本地生活推荐约束审计员。你会看到一个用户查询、人工约束清单，以及两个匿名系统A/B的推荐和工具证据。reference answer/点击POI已被移除，不得猜测或使用它。

逐个推荐独立判断：
- qualified：query中的全部硬约束和软偏好均有该POI自身工具证据支持。
- acceptable：全部硬约束均满足；仅有软偏好未满足或无法验证，并且该推荐在constraint_gaps中明确披露了该缺口。
- unacceptable：任一硬约束不满足或缺乏证据；软偏好缺失却未在constraint_gaps披露；理由扭曲或编造工具事实；证据属于其他POI。

规则：
1. 以query为最高依据；人工约束若与query冲突，以query为准。
2. 人工约束的h=刚性必须满足且有证据，披露硬缺口也仍是不合格。h=柔性可以放松，但必须在该推荐自己的constraint_gaps中披露。
3. 工具未返回字段表示未知，不代表满足。实时状态不能由静态营业时间或历史评论代替。
4. 只用每个推荐附带的observed_facts与cited_evidence，不把reason或constraint_gaps本身当证据。
5. 不因为某POI不是点击目标而判错；只判断它自身是否满足query。
6. 对A/B使用完全相同标准，不猜模型来源，不比较推荐数量。

只输出合法JSON：
{"A":[{"rank":1,"verdict":"qualified|acceptable|unacceptable","reason":"简短依据"}],"B":[...]}
数组长度和rank必须与输入推荐逐一对应。'''

COMPACT_FIELDS = (
    "poiid", "name", "poi_name", "evidence_id", "category", "category_path",
    "address", "district", "business_area", "loc", "distance_m", "rating",
    "average_cost", "star", "open_time", "tags", "tickets",
    "mall_hotel_hours_flag", "parking_spaces", "comment_count", "content",
    "open_tag", "open_tag_term", "open_tag_detail", "created_at",
)


def load_run(path: Path) -> Dict[str, Dict[str, Any]]:
    records, _, parse_errors = dedupe_records(read_jsonl(path))
    if parse_errors:
        raise ValueError("%s 存在无法解析的行：%s" % (path, dict(parse_errors)))
    return {record["task_id"]: record for record in records}


def compact_item(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: item[key]
        for key in COMPACT_FIELDS
        if key in item and item[key] not in (None, "", [], {})
    }


def evidence_catalog(record: Dict[str, Any]) -> Tuple[Dict[str, List], Dict[str, Dict]]:
    """Facts the agent actually saw, indexed by poiid and by evidence_id."""
    by_poi: Dict[str, List] = {}
    by_evidence: Dict[str, Dict] = {}
    seen_poi: Dict[str, set] = {}
    for event in record.get("trajectory") or []:
        if event.get("event") != "tool_calls":
            continue
        for call in event.get("calls") or []:
            observation = call.get("observation")
            if not isinstance(observation, dict):
                continue
            data = observation.get("data")
            if not isinstance(data, dict):
                continue
            for key in ("results", "notes"):
                for item in data.get(key) or []:
                    if not isinstance(item, dict):
                        continue
                    compact = compact_item(item)
                    evidence_id = item.get("evidence_id")
                    if isinstance(evidence_id, str):
                        by_evidence[evidence_id] = compact
                    poiid = item.get("poiid")
                    if isinstance(poiid, str):
                        blob = json.dumps(compact, ensure_ascii=False, sort_keys=True)
                        if blob not in seen_poi.setdefault(poiid, set()):
                            seen_poi[poiid].add(blob)
                            by_poi.setdefault(poiid, []).append(compact)
    return by_poi, by_evidence


def compact_answer(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    by_poi, by_evidence = evidence_catalog(record)
    output = []
    recommendations = (record.get("answer") or {}).get("recommendations") or []
    for rank, recommendation in enumerate(recommendations, 1):
        poiid = recommendation.get("poiid")
        output.append(
            {
                "rank": rank,
                "poiid": poiid,
                "name": recommendation.get("name"),
                "reason": recommendation.get("reason"),
                "constraint_gaps": recommendation.get("constraint_gaps") or [],
                "evidence_ids": recommendation.get("evidence") or [],
                "cited_evidence": [
                    by_evidence[evidence_id]
                    for evidence_id in recommendation.get("evidence") or []
                    if evidence_id in by_evidence
                ],
                "observed_facts": (by_poi.get(poiid) or [])[:12],
            }
        )
    return output


def parse_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text[text.find("\n") + 1 : text.rfind("```")].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def validate(result: Any, expected: Dict[str, int]) -> None:
    if not isinstance(result, dict):
        raise ValueError("judge result is not object")
    for arm, count in expected.items():
        rows = result.get(arm)
        if not isinstance(rows, list) or len(rows) != count:
            raise ValueError(
                "%s count mismatch: expected %d got %r"
                % (arm, count, len(rows) if isinstance(rows, list) else None)
            )
        for index, row in enumerate(rows, 1):
            if (
                not isinstance(row, dict)
                or row.get("rank") != index
                or row.get("verdict") not in VERDICTS
            ):
                raise ValueError("%s invalid row %d: %r" % (arm, index, row))
            if not isinstance(row.get("reason"), str) or not row["reason"].strip():
                raise ValueError("%s empty reason %d" % (arm, index))


def call_judge(args, task_id: str, payload: Dict[str, Any], expected: Dict[str, int]) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    last: Exception | None = None
    for _ in range(args.judge_retries):
        body = {
            "model": args.judge_model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": args.judge_max_tokens,
            "chat_template_kwargs": {"enable_thinking": bool(args.judge_thinking)},
        }
        request = urllib.request.Request(
            args.judge_url,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=args.judge_timeout) as response:
                raw = json.loads(response.read().decode("utf-8"))
            text = raw["choices"][0]["message"]["content"]
            result = parse_json(text)
            validate(result, expected)
            return {
                "task_id": task_id,
                "status": "completed",
                "judgment": result,
                "usage": raw.get("usage") or {},
            }
        except Exception as exc:  # retry with the schema error appended
            last = exc
            messages.append(
                {
                    "role": "user",
                    "content": "上一输出无效：%s。只返回符合schema的JSON。" % exc,
                }
            )
    return {
        "task_id": task_id,
        "status": "failed",
        "error": {"type": type(last).__name__, "message": str(last)},
    }


def blind_mapping(task_id: str, label_a: str, label_b: str) -> Dict[str, str]:
    """Stable per-task A/B assignment; the judge only ever sees A and B."""
    a_first = int(hashlib.sha256(task_id.encode("utf-8")).hexdigest(), 16) % 2 == 0
    return {"A": label_a if a_first else label_b, "B": label_b if a_first else label_a}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True, help="带人工约束标注的输入json")
    parser.add_argument("--arm-a", required=True, help="A臂 run 输出 jsonl")
    parser.add_argument("--arm-b", required=True, help="B臂 run 输出 jsonl")
    parser.add_argument("--label-a", default="arm_a")
    parser.add_argument("--label-b", default="arm_b")
    parser.add_argument("--output", required=True)
    parser.add_argument("--judge-url", default="http://127.0.0.1:8001/v1/chat/completions")
    parser.add_argument("--judge-model", default="Qwen3.6-35B-A3B")
    parser.add_argument("--judge-thinking", type=int, default=1)
    parser.add_argument("--judge-max-tokens", type=int, default=8192)
    parser.add_argument("--judge-timeout", type=int, default=900)
    parser.add_argument("--judge-retries", type=int, default=3)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--fresh", action="store_true", help="忽略已有输出，重新审计")
    args = parser.parse_args()

    source_records = json.load(open(args.tasks, encoding="utf-8"))["records"]
    source = {record["id"]: record for record in source_records}
    arm_a = load_run(Path(args.arm_a))
    arm_b = load_run(Path(args.arm_b))

    order = [task_id for task_id in source if task_id in arm_a and task_id in arm_b]
    skipped = len(source) - len(order)
    if args.limit is not None:
        order = order[: args.limit]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    done: set = set()
    if output.exists() and not args.fresh:
        for line in output.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("status") == "completed":
                done.add(record["task_id"])
    pending = [task_id for task_id in order if task_id not in done]

    def prepare(task_id: str):
        mapping = blind_mapping(task_id, args.label_a, args.label_b)
        answers = {
            args.label_a: compact_answer(arm_a[task_id]),
            args.label_b: compact_answer(arm_b[task_id]),
        }
        record = source[task_id]
        payload = {
            "query": record.get("query"),
            "context": record.get("context"),
            "constraints": record.get("constraints") or [],
            "structure": record.get("structure") or {},
            "system_A_recommendations": answers[mapping["A"]],
            "system_B_recommendations": answers[mapping["B"]],
        }
        expected = {
            "A": len(answers[mapping["A"]]),
            "B": len(answers[mapping["B"]]),
        }
        return payload, expected, mapping

    print(
        json.dumps(
            {
                "tasks": len(source),
                "pairs": len(order),
                "skipped_missing_arm": skipped,
                "already_judged": len(done),
                "pending": len(pending),
                "judge": args.judge_model,
                "workers": args.workers,
                "output": str(output.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )

    lock = threading.Lock()
    counts = {"completed": 0, "failed": 0}
    with output.open("w" if args.fresh else "a", encoding="utf-8") as handle:
        def work(task_id: str):
            payload, expected, mapping = prepare(task_id)
            result = call_judge(args, task_id, payload, expected)
            result["blind_mapping"] = mapping
            return result

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(work, task_id): task_id for task_id in pending}
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                with lock:
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    handle.flush()
                    counts[result["status"] if result["status"] in counts else "failed"] += 1
                    finished = counts["completed"] + counts["failed"]
                print(
                    "%d/%d %s %s" % (finished, len(pending), result["task_id"], result["status"]),
                    flush=True,
                )
    print(json.dumps(counts, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
