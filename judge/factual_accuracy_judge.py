"""Judge query-level factual accuracy of an agent run with an OpenAI-compatible LLM."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "scripts"))
sys.path.insert(0, str(_HERE))

from finalize_run import dedupe_records, read_jsonl  # noqa: E402
from paired_acceptability_judge import compact_answer, evidence_catalog, parse_json  # noqa: E402

SYSTEM = '''你是严格的本地生活答案事实准确性审计员。你会看到一条用户 query、上下文，以及匿名系统的完整答案和该系统实际看到的工具证据。以 query 为单位判断答案中是否存在事实性错误。

定义：
- factual_correct=1：答案中的所有可核验事实陈述都与该答案所附的工具证据一致，没有错误数字、错误实体属性、跨 POI 张冠李戴或无证据编造的具体事实。
- factual_correct=0：至少存在一处事实性错误。

判定规则：
1. 只判事实准确性，不判推荐是否满足 query 的约束，不判推荐质量、丰富度或主观审美。POI 不符合用户需求，但答案对它的描述完全真实时，不算事实错误。
2. 只用输入中每个推荐的 cited_evidence/observed_facts 核验该推荐的 reason；summary/notes 还可使用 all_observed_evidence（系统在完整工具轨迹中看过的全部候选与笔记）核验，包括未被最终推荐的候选。不得使用常识、reference answer、点击目标、模型身份或外部知识补证。
3. 数字必须一致，包括价格、人均、评分、距离、营业时间、评论数等。允许不改变含义的合理近似或四舍五入，例如 49.8 元写“约50元”；50 元写成60元属于 factual error。
4. 实体属性必须一致，包括名称、地址、区域、类别、服务、设施、商品、菜品、是否24小时、是否当前营业等。父类别不能证明具体服务；历史评论不能证明实时库存、排队、可约或当前营业状态。
5. 证据属于其他 POI 却用于描述当前推荐，属于 cross_poi_attribution 错误。答案把工具未提供的具体可验证事实说成确定事实，属于 unsupported_claim。
6. subjective/概括性表达（如“值得考虑”“氛围不错”）只有在明显扭曲证据或与证据冲突时才算错误。单纯措辞夸张但不形成具体可验证事实时，不要过度判错。
7. constraint_gaps 是系统自述，不是事实证据；但准确披露不确定性不会造成事实错误。遗漏 query 约束、推荐过少、拒答或 no_exact_match 是否合理，不属于本任务，除非答案同时陈述了与证据矛盾的事实。
8. 每个 error 必须指出 answer_claim、evidence_fact 和 error_type。error_type 只能是：numeric_mismatch、attribute_mismatch、temporal_mismatch、cross_poi_attribution、unsupported_claim、summary_mismatch。
9. 没有事实错误时 errors 必须为空数组；存在任一错误时 factual_correct 必须为0。不要把“证据充分但不满足 query”误判成事实错误。

只输出合法 JSON：
{"factual_correct":1,"errors":[],"reason":"简短总体结论"}
或
{"factual_correct":0,"errors":[{"rank":1,"error_type":"numeric_mismatch","answer_claim":"人均60元","evidence_fact":"average_cost=50","explanation":"答案数字与工具字段不一致"}],"reason":"简短总体结论"}
summary/notes 层面的错误 rank 填0。'''


def load_run(path: Path) -> Dict[str, Dict[str, Any]]:
    records, _, parse_errors = dedupe_records(read_jsonl(path))
    if parse_errors:
        raise ValueError(f"{path} has unparseable lines: {dict(parse_errors)}")
    return {r["task_id"]: r for r in records if r.get("status") == "completed"}


def validate(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict) or value.get("factual_correct") not in (0, 1):
        raise ValueError("invalid factual_correct")
    errors = value.get("errors")
    if not isinstance(errors, list):
        raise ValueError("errors must be list")
    if value["factual_correct"] == 1 and errors:
        raise ValueError("factual_correct=1 requires empty errors")
    if value["factual_correct"] == 0 and not errors:
        raise ValueError("factual_correct=0 requires nonempty errors")
    allowed = {
        "numeric_mismatch", "attribute_mismatch", "temporal_mismatch",
        "cross_poi_attribution", "unsupported_claim", "summary_mismatch",
    }
    for error in errors:
        if not isinstance(error, dict) or error.get("error_type") not in allowed:
            raise ValueError("invalid error")
        if not isinstance(error.get("rank"), int) or error["rank"] < 0:
            raise ValueError("invalid error rank")
        for key in ("answer_claim", "evidence_fact", "explanation"):
            if not isinstance(error.get(key), str) or not error[key].strip():
                raise ValueError(f"invalid {key}")
    if not isinstance(value.get("reason"), str) or not value["reason"].strip():
        raise ValueError("invalid reason")
    return value


def call_judge(args: argparse.Namespace, api_key: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    last: Exception | None = None
    for attempt in range(args.judge_retries):
        body = {
            "model": args.judge_model,
            "messages": messages,
            "temperature": args.judge_temperature,
            "max_tokens": args.judge_max_tokens,
        }
        request = urllib.request.Request(
            args.judge_url,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + api_key},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=args.judge_timeout) as response:
                raw = json.loads(response.read().decode("utf-8"))
            text = raw["choices"][0]["message"]["content"]
            judgment = validate(parse_json(text))
            return {"status": "completed", "judgment": judgment, "usage": raw.get("usage") or {}}
        except urllib.error.HTTPError as exc:
            last = exc
            if attempt + 1 < args.judge_retries:
                time.sleep(min(2 ** attempt, 8))
        except Exception as exc:
            last = exc
            if attempt + 1 < args.judge_retries:
                messages.append({"role": "user", "content": f"上一输出无效：{exc}。只返回符合schema的JSON。"})
    return {"status": "failed", "error": {"type": type(last).__name__, "message": str(last)}}


def build_payload(record: Dict[str, Any], task_id: str) -> Dict[str, Any]:
    answer = record.get("answer") or {}
    _, by_evidence = evidence_catalog(record)
    return {
        "task_id": task_id,
        "query": (record.get("input") or {}).get("query"),
        "context": {k: v for k, v in (record.get("input") or {}).items() if k != "query"},
        "answer": {
            "summary": answer.get("summary"),
            "recommendations": compact_answer(record),
            "no_exact_match": answer.get("no_exact_match"),
            "notes": answer.get("notes"),
        },
        "all_observed_evidence": list(by_evidence.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--judge-url", default=os.environ.get("JUDGE_URL") or os.environ.get("ATLAS_JUDGE_URL"),
                        help="Judge endpoint; defaults to $JUDGE_URL / $ATLAS_JUDGE_URL. Required unless --dry-run.")
    parser.add_argument("--judge-model", default="deepseek-v4-pro")
    parser.add_argument("--judge-api-key-env", default="ATLAS_JUDGE_API_KEY")
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--judge-max-tokens", type=int, default=4096)
    parser.add_argument("--judge-timeout", type=int, default=900)
    parser.add_argument("--judge-retries", type=int, default=5)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="只显示待评分输入，不调用API或写入结果")
    args = parser.parse_args()

    if not args.dry_run and not args.judge_url:
        parser.error("--judge-url is required unless --dry-run is used")
    run = load_run(Path(args.run))
    order = list(run.keys())
    if args.limit is not None:
        order = order[:args.limit]
    if args.dry_run:
        for task_id in order:
            print(json.dumps(build_payload(run[task_id], task_id), ensure_ascii=False, indent=2))
        return
    api_key = os.environ.get(args.judge_api_key_env, "")
    if not api_key:
        raise SystemExit(f"{args.judge_api_key_env} is not set")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if output.exists() and not args.fresh:
        for line in read_jsonl(output):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("status") == "completed":
                done.add(r["task_id"])
    pending = [task_id for task_id in order if task_id not in done]
    print(json.dumps({"selected": len(order), "pending": len(pending), "model": args.judge_model}, ensure_ascii=False), flush=True)

    def work(task_id: str) -> Dict[str, Any]:
        payload = build_payload(run[task_id], task_id)
        result = call_judge(args, api_key, payload)
        result.update({"task_id": task_id, "label": args.label})
        return result

    lock = threading.Lock()
    counts = Counter()
    with output.open("w" if args.fresh else "a", encoding="utf-8") as handle:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(work, task_id): task_id for task_id in pending}
            for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
                result = future.result()
                with lock:
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    handle.flush()
                if result["status"] == "completed":
                    counts[result["judgment"]["factual_correct"]] += 1
                else:
                    counts["failed"] += 1
                print(f"{index}/{len(pending)} {result['task_id']} {result['status']}", flush=True)
    print(json.dumps({"factual_correct": counts[1], "factual_error": counts[0], "failed": counts["failed"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
