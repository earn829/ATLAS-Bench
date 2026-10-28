---
name: atlas-benchmark-eval
description: Reproduce and operate ATLAS-Bench 1200-query model evaluations in this v6 repository. Use for local vLLM or Hosted runs, thinking-mode comparisons, smoke tests, concurrency selection, resumable execution, failure recovery, finalization, programmatic-judge summaries, prompt caching, and fairness reporting.
argument-hint: <profile> [smoke|full|retry|finalize|status]
---

# ATLAS Benchmark Evaluation

Run every model through the same v6 ReAct agent and preserve enough configuration metadata for reproducible comparisons.

## Invariants

Run from the repository root. Do not change these between compared runs unless the experiment explicitly studies that variable:

- Input: `../query/subset_1200_agent_input.json` (1200 records).
- System prompt: `prompts/react_system_prompt.md`.
- Maximum 5 tool rounds, 2 calls per round, 10 calls total.
- `default_top_k=10`, `max_top_k=20`, `vector_candidate_k=100`.
- `max_observation_chars=32000`, recommendation prompt `1--10`.
- Retrieval runs on GPU0; local generation runs on GPU1.
- Temperature is 0.1 unless the provider rejects it. Omit it rather than silently substituting another value.
- Do not feed `reasoning` or `reasoning_content` back into conversation history; only assistant `content` belongs in history.
- Keep POI/name validation as a hard generation constraint. Treat no-exact-match and evidence checks as metrics, not regeneration triggers.

Record every necessary exception in the final config JSON: output budget changes, repetition penalty, reasoning effort, temperature omission, retry workers, and cache behavior.

## Choose a profile

Read `references/model-profiles.md` before running a known model. Use its exact model ID and thinking controls. Never infer a control parameter from another provider.

Use the bundled runner for known profiles:

```bash
conda activate <your-env>   # or activate your virtualenv; it only needs the repository's dependencies
python .qoder/skills/atlas-benchmark-eval/scripts/run_profile.py <profile> smoke
python .qoder/skills/atlas-benchmark-eval/scripts/run_profile.py <profile> full
python .qoder/skills/atlas-benchmark-eval/scripts/run_profile.py <profile> status
python .qoder/skills/atlas-benchmark-eval/scripts/run_profile.py <profile> retry
python .qoder/skills/atlas-benchmark-eval/scripts/run_profile.py <profile> finalize
```

Pass `--input PATH` when the 1200-query file is elsewhere. No endpoint or credential is hardcoded; both profile families read the environment at call time:

- Hosted profiles require `ATLAS_HOSTED_API_KEY` plus `ATLAS_HOSTED_API_BASE` (an OpenAI-compatible base) or `ATLAS_HOSTED_API_URL` (a full POST URL). Never print or persist the key.
- Local profiles default to a vLLM server at `http://127.0.0.1:8001/v1` with the placeholder key `local-vllm`; override with `ATLAS_LOCAL_API_BASE` and `ATLAS_LOCAL_API_KEY` when the server listens elsewhere. The expected model must already be served there.

Judge runs are configured the same way: `--judge-url` or `JUDGE_URL` / `ATLAS_JUDGE_URL` for the endpoint and `ATLAS_JUDGE_API_KEY` (or `--judge-api-key-env NAME`) for the key. `--dry-run` needs neither.

## Workflow

### 1. Preflight

1. Verify the input contains exactly 1200 records.
2. Verify the SQLite/FAISS indexes exist under `data/indexes/`, and run `python3 scripts/fetch_models.py --check` — the Qwen3 embedding/reranker weights are not shipped and must be fetched at the pinned revisions before any retrieval run.
3. For local inference, verify `/v1/models` returns the profile's model and GPU1 owns the serving process. Keep GPU0 available for retrieval.
4. For hosted inference, make one harmless request to verify the exact model ID and supported parameters.
5. Do not start a second full run using the same output path.

### 2. Smoke test

Run `smoke` before every new model or mode. It processes 10 tasks with at most 4 workers and writes to an isolated `_smoke_*` directory.

Require:

- 10 unique records and no parse errors.
- No unexpected protocol or API failures.
- Thinking state matches the profile: zero reasoning tokens for non-thinking; nonzero tokens or provider-specific hidden-thinking evidence for thinking.
- No completion reaches the output-token ceiling.
- Prompt plus completion stays below the deployed context limit.

A programmatic-judge miss is a quality observation, not a smoke-test infrastructure failure.

### 3. Full run

Run `full` only after smoke passes. The command appends durable JSONL records and flushes each completed task. Re-running the same command resumes automatically and skips only completed task IDs.

Never use `--fresh` on a valuable run. Never run two writers against the same JSONL.

### 4. Monitor

Use `status` for deduplicated counts. Monitor:

- unique/completed/failed tasks;
- failure types;
- reasoning/completion ratio;
- output-budget hits;
- recommendation mean;
- process liveness.

Do not treat raw line count as task count after retries.

### 5. Recover failures

Preserve the main run and append retries to the same JSONL.

1. Retry transient HTTP, disconnect, and protocol failures with the same model settings and 1--3 workers.
2. For `OutputBudgetExhausted`, retry only failed tasks with a larger output budget (normally 16384).
3. If repeated reasoning loops persist and the provider supports effort controls, use the lowest supported effort only for those failed tasks and record the mixed configuration.
4. Do not change temperature merely to rescue failures; that changes the experiment.
5. Stop repeated identical retries after three failed task-level attempts unless a concrete parameter change addresses the cause.

### 6. Finalize

Run `finalize` after all practical recovery. It calls `scripts/finalize_run.py`, prefers a completed attempt, preserves the latest failure otherwise, orders records by the source task file, and creates:

- `subset_1200.final.jsonl`
- `subset_1200.final.jsonl.summary.json`

Verify `records=1200`, `missing=0`, and report any retained failures honestly.

### 7. Report

Always report:

- completion/failure counts and retry count;
- answer pass rate and no-exact-match consistency;
- evidence existence and ownership accuracy;
- recommendation distribution/mean;
- prompt, completion, reasoning and cached tokens per query;
- tool calls and latency per query;
- all configuration differences affecting fairness.

For paired mode comparisons, compare identical task IDs and report pass/fail transitions, not only aggregate percentages.

These are run-level metrics. CSR, CR and Success@k come from the blind constraint audit in the `atlas-qoder-audit` skill, run on the finalized JSONL this skill produces.

## Model-specific cautions

- Qwen local thinking may loop. Use the profile's repetition penalty and retry only budget-exhausted tasks at 16384.
- GLM default behavior is Max-like and may exhaust 16384. `reasoning_effort=low` is the nearest available rescue; `none` is rejected.
- Kimi-K3 rejects temperature 0.1, so omit temperature. Its default behavior is Max-like.
- GPT Luna models support `none/low/medium/high/xhigh`, hide reasoning text, and accept only default temperature 1; omit temperature.
- Claude Bedrock supports Anthropic prompt caching through `cache_control: {type: ephemeral}`. Enable only `LOCAL_LIFE_CACHE_SYSTEM_PROMPT=true`, use one worker under rate limiting, and increase request retries rather than concurrency.
- Gemini hides reasoning text. Use usage reasoning tokens when provided.

## Resources

- `references/model-profiles.md`: tested model IDs and exact run profiles.
- `scripts/run_profile.py`: deterministic smoke/full/retry/finalize/status wrapper around the repository scripts.
- `.qoder/skills/atlas-qoder-audit/`: the blind constraint audit that turns a finalized run into CSR, CR and Success@k.
