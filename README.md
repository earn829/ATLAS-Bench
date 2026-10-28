<div align="center">

<h1>ATLAS-Bench</h1>

<p><strong>An Agentic Tool-Use Benchmark for Local Area Search</strong></p>

<p>📍 Context-aware search &nbsp;·&nbsp; 🔎 Evidence-supported recommendations</p>

<p>
  <a href="#benchmark-overview"><img src="assets/badges/queries.svg" alt="1,200 queries" height="26"></a>
  <a href="#benchmark-overview"><img src="assets/badges/regions.svg" alt="3 regions" height="26"></a>
  <a href="#running-a-model"><img src="assets/badges/models.svg" alt="11 evaluated models" height="26"></a>
  <a href="#running-a-model"><img src="assets/badges/configurations.svg" alt="14 evaluated configurations" height="26"></a>
</p>

<p>
  <a href="#benchmark-overview">🧭 Benchmark Overview</a> &nbsp;·&nbsp;
  <a href="#quickstart">🚀 Quickstart</a> &nbsp;·&nbsp;
  <a href="#evaluation">📊 Evaluation</a>
</p>

</div>

> **Finding a relevant place is not enough.**
>
> An agent must verify that it meets the user's requirements.

**ATLAS-Bench** evaluates agentic local search under spatial, temporal, and user-specified
constraints. It contains **1,200 log-grounded synthetic queries** across Chaoyang (Beijing),
Huangpu (Shanghai), and Dabancheng (Urumqi). Queries are expanded from real search events using
POI attributes, review evidence, and platform-derived user personas, then validated through
iterative assertion checks and cross-model review.

<table>
<tr>
<td width="50%" valign="top">

<h3>🧩 Requirements with evidence</h3>

<p>Each requirement is annotated along two independent dimensions—<strong>hard vs. soft</strong> and <strong>objective vs. subjective</strong>—with constraint relations, supporting evidence, and verification criteria.</p>

<p><strong>Hard constraints</strong> must all be supported. Any unmet or unverifiable <strong>soft preference</strong> must be explicitly disclosed.</p>

<p>The reference POI used during synthesis is <strong>not the only valid answer</strong>.</p>

</td>
<td width="50%" valign="top">

<h3>📦 A controlled search sandbox</h3>

<p><strong>ATLASBox</strong> provides standardized search tools and a fixed POI knowledge base integrating structured attributes, reviews, and place-related notes.</p>

<p><strong>Agents see</strong><br>Query, timestamp, and user location only.</p>

<p><strong>Withheld from agents</strong><br>Personas and reference-target annotations.</p>

</td>
</tr>
</table>

<p align="center">
  Jointly evaluated<br>
  <strong>Answer quality &nbsp;·&nbsp; Trajectory quality &nbsp;·&nbsp; Execution cost</strong><br>
  Distinguishing reliable recommendations from query-level success.
</p>

---

<a id="table-of-contents"></a>

## 📑 Table of Contents

- [🧭 Benchmark Overview](#benchmark-overview)
- [📂 Repository Structure](#repository-structure)
- [🛠️ Installation](#installation)
- [🚀 Quickstart](#quickstart)
- [🤖 Running a Model](#running-a-model)
- [📊 Evaluation](#evaluation)
- [🧪 Tests](#tests)
- [🔁 Reproducibility](#reproducibility)

---

<a id="benchmark-overview"></a>

## 🧭 Benchmark Overview

### 📚 Data

| | |
|---|---|
| **Queries** | 1,200 log-grounded synthetic queries with timestamp and user-location context (400 each for Chaoyang, Beijing; Huangpu, Shanghai; and Dabancheng, Urumqi) |
| **Annotations** | Hard/soft and objective/subjective constraint labels, constraint relations, supporting evidence, verification criteria, and satisfiability labels; 1,004 queries are satisfiable |
| **Knowledge base** | 425,244 POIs (structured fields + geo) and 260,757 user notes, frozen into SQLite (FTS5 + RTree) and FAISS indexes |
| **Retrieval** | Qwen3-Embedding-0.6B (1024-d) → FAISS exact inner product (top-100) → Qwen3-Reranker-0.6B; identical for every model under test |

### 🤖 Agent

A bounded ReAct loop with four tools (`search_pois`, `search_notes`, `get_poi_notes`,
`search_nearby` — see [TOOLS.md](TOOLS.md)). Hard limits: **5 tool rounds, ≤2 parallel calls per
round, ≤10 calls total**. The final answer is a ranked list of 1–10 POIs, each with its own
rationale, cited evidence IDs, and disclosed `constraint_gaps`. Recommended POI IDs and names are
validated against tool observations at generation time — hallucinated merchants are rejected.

### 📏 Metrics (12 columns)

| Group | Metric | Source |
|---|---|---|
| **Answer quality** | CSR / CR / Success@k | Blind constraint audit against `judge/RUBRIC.md` |
| | Recs/query | Run telemetry |
| | Factual accuracy | Programmatic grounding judge **∧** LLM factuality judge |
| **Trajectory quality** | Intent understanding | LLM judge |
| | Tool-use appropriateness | LLM judge |
| | Evidence sufficiency | LLM judge |
| | Strategy adaptation | LLM judge |
| **Efficiency** | Tokens / Tool calls / Rounds per query | Run telemetry |

---

<a id="repository-structure"></a>

## 📂 Repository Structure

```
├── query/                     # Benchmark inputs
│   ├── subset_1200_agent_input.json   # What the agent sees (id/query/context only)
│   └── subset_1200.json               # Full annotations (judges only — never fed to the agent)
├── data/
│   ├── indexes/               # Frozen retrieval corpus (SQLite + FAISS + manifests, via Git LFS)
│   └── models/                # Qwen3 embedding/reranker weights — downloaded, not committed
├── local_life_agent/          # ReAct controller, protocol, retrieval stack, tool registry
├── prompts/                   # System prompt (part of the frozen experimental setup)
├── scripts/                   # Runners (agent, finalize, compare), index builders, judge drivers
├── judge/                     # Scoring chain: RUBRIC.md + all judge/aggregation scripts
├── tests/                     # 104 offline contract tests (no GPU, no network, no API keys)
└── .qoder/skills/             # Step-by-step playbooks: running models & the blind audit
```

---

<a id="installation"></a>

## 🛠️ Installation

### 🐍 1. Python environment

Python ≥ 3.9 (3.10+ only if you plan to rebuild indexes with MLX on Apple Silicon).
The requirements include FAISS, sentence-transformers, and transformers.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-vector.txt
```

### 🗂️ 2. Retrieval indexes

The frozen SQLite + FAISS indexes (~4.1 GB) ship with the repository via Git LFS:

```bash
git lfs install && git lfs pull
```

### 🧠 3. Retrieval models

Fetch the Qwen3 embedding/reranker weights (~2.3 GB) into `data/models/`, then verify that both
match the pinned revisions used to build the indexes:

```bash
python3 scripts/fetch_models.py
python3 scripts/fetch_models.py --check
```

> The pins matter: FAISS vectors and query embeddings must come from the same model revision,
> otherwise retrieval silently degrades. Details in [`data/README.md`](data/README.md).

### ✅ 4. Verify

Run the offline tests, check retrieval-model loading and numerics, then try a POI search:

```bash
python3 -m unittest discover -s tests -v
python3 scripts/check_neural_models.py
python3 scripts/run_tool.py search_pois '{"query":"咖啡厅","top_k":3}'
```

Disk: ~4.1 GB (indexes) + ~2.3 GB (models). GPU is recommended for retrieval but not required
(`LOCAL_LIFE_VECTOR_DEVICE=auto|cuda|mps|cpu`).

---

<a id="quickstart"></a>

## 🚀 Quickstart

Try one tool call against the frozen corpus:

```bash
python3 scripts/run_tool.py search_pois '{"query":"咖啡厅","top_k":3}'
python3 scripts/run_tool.py search_nearby '{"query":"咖啡厅","loc":{"lng":116.46,"lat":40.0},"distance_m":2000,"top_k":5}'
```

Run the agent on a single query (after configuring a model, next section):

```bash
python3 scripts/run_agent.py --limit 1 --workers 1 --output outputs/smoke_one.jsonl
```

---

<a id="running-a-model"></a>

## 🤖 Running a Model

The agent talks to any **OpenAI-compatible HTTP endpoint**. Configure via
environment variables (full list with comments: [`.env.example`](.env.example)). Keys are only
ever read from the environment; nothing is hardcoded.

### 🖥️ Option A — Local model (vLLM)

Start the model server in a separate terminal:

```bash
vllm serve /path/to/model --port 8001
```

Point the agent at the server. `local-vllm` is a placeholder key; vLLM ignores it.
Set `LOCAL_LIFE_ENABLE_THINKING` to `true` or `false` through `chat_template_kwargs`.

```bash
export LOCAL_LIFE_API_BASE='http://127.0.0.1:8001/v1'
export LOCAL_LIFE_API_KEY='local-vllm'
export LOCAL_LIFE_MODEL='<served-model-name>'
export LOCAL_LIFE_ENABLE_THINKING='false'
export LOCAL_LIFE_THINKING_FIELD='chat_template_kwargs'
```

### ☁️ Option B — API model (OpenAI-compatible gateway)

```bash
export LOCAL_LIFE_API_BASE='https://your-host.example/v1'
export LOCAL_LIFE_API_KEY='<your-api-key>'
export LOCAL_LIFE_MODEL='<model-id>'
export LOCAL_LIFE_API_STYLE='chat_completions'
```

Use `LOCAL_LIFE_API_URL` for a full POST URL. Other API styles include `responses` and
`anthropic` (auto-detected for Claude).

`OPENAI_BASE_URL/OPENAI_API_KEY/OPENAI_MODEL_NAME` and `API_URL/API_KEY/MODEL_NAME` are accepted
as aliases. Models that reject `temperature` (some Claude/GPT endpoints): set
`LOCAL_LIFE_OMIT_TEMPERATURE=true`. Reasoning effort, repetition penalty, output budget, caching —
all documented in `.env.example`.

### ▶️ Smoke → Full → Resume → Finalize

**Smoke test.** Run 10 queries and inspect the trajectories before a full run:

```bash
python3 scripts/run_agent.py --limit 10 --workers 1 --fresh --output outputs/smoke.jsonl
```

**Full run.** Execute all 1,200 queries from `query/subset_1200_agent_input.json`.
To resume after an interruption, repeat the same command; completed task IDs are skipped.

```bash
python3 scripts/run_agent.py --workers 4 --output outputs/my_model.jsonl
```

**Finalize.** Consolidate the output into one record per task and generate a summary:

```bash
python3 scripts/finalize_run.py --input outputs/my_model.jsonl
```

Each JSONL record contains the input whitelist, final answer, `programmatic_judge` result, the
full tool-observation trajectory, protocol-correction events, latency and token usage.
Compare two runs on identical task IDs with `scripts/compare_runs.py`.

> Model configurations (exact model IDs, thinking controls, budgets, retry
> policy) are recorded in [`model-profiles.md`](.qoder/skills/atlas-benchmark-eval/references/model-profiles.md), with a deterministic runner
> `run_profile.py <profile> smoke|full|retry|finalize|status`.

---

<a id="evaluation"></a>

## 📊 Evaluation

All scoring code lives in [`judge/`](judge/); every metric below is regenerated from your own run
JSONL — no precomputed results are shipped.

### 🎯 1. Answer quality — CSR, CR, Success@k (blind constraint audit)

A recommendation-level audit against the annotated constraints and supporting evidence, following
[`judge/RUBRIC.md`](judge/RUBRIC.md) (verdicts: *qualified / acceptable / unacceptable*).
Inputs are **blind**: model identity, reference answers and click targets are stripped.

Generate the audit inputs:

```bash
python3 judge/build_qoder_judge_inputs.py --run outputs/my_model.final.jsonl --label my_model
```

Adjudicate each `judge/my_model/inputs/chunk_*.json` using `judge/RUBRIC.md` and save the
results to `judge/my_model/verdicts/chunk_*.jsonl`. Then compute CSR, CR, Success@k, and
Recs/query with contract validation:

```bash
python3 .qoder/skills/atlas-qoder-audit/scripts/score_verdicts.py my_model
```

The full procedure (chunk-by-chunk adjudication, anti-bias rules, published formulas) is packaged
as the [`atlas-qoder-audit`](.qoder/skills/atlas-qoder-audit/SKILL.md) skill.

### ✅ 2. Factual accuracy

`programmatic_judge` (deterministic evidence-ownership checks, embedded in every run record)
**AND** an LLM factuality judge:

```bash
JUDGE_URL='https://your-judge-host.example/v1/chat' ATLAS_JUDGE_API_KEY=... \
  bash scripts/run_all_factual_accuracy.sh --run outputs/my_model.final.jsonl --label my_model --output judge/factual/my_model.jsonl
python3 judge/combine_passrate_factual.py
```

### 🧵 3. Trajectory quality — four LLM judges

Intent understanding, tool-use appropriateness, evidence sufficiency, strategy adaptation.
Each judge has an anti-inflation rubric with hard score caps enforced by a schema validator:

```bash
export JUDGE_URL='https://your-judge-host.example/v1/chat'
export ATLAS_JUDGE_API_KEY=...
bash scripts/run_all_intent_understanding.sh --run outputs/my_model.final.jsonl --label my_model --output ...
python3 judge/summarize_intent_understanding.py
```

Run the other dimensions with `scripts/run_all_tool_selection.sh`,
`scripts/run_all_evidence_sufficiency.sh`, and `scripts/run_all_tool_flexibility.sh`.
Each dimension has a corresponding `judge/summarize_*.py` script that generates its leaderboard.

Every driver supports `--dry-run` (prints exact judge payloads, no API calls, no credentials
needed) and `smoke` mode (15 queries).

### ⚡ 4. Efficiency

Tokens / tool calls / rounds per query come from run telemetry — `scripts/finalize_run.py`'s
summary reports them directly; no judge involved.

---

<a id="tests"></a>

## 🧪 Tests

```bash
python3 -m unittest discover -s tests -v
```

104 offline contract tests: retrieval semantics (real FTS/RTree on tiny fixtures, fake vectors),
tool filtering order, input whitelisting, hallucination guards, run/finalize/compare pipelines,
judge payload builders, scoring formulas, and full relocation/portability checks. No GPU, network,
or API keys required.

---

<a id="reproducibility"></a>

## 🔁 Reproducibility

Keep the following settings fixed across compared runs:

| Component | Shared configuration |
|---|---|
| **System prompt** | [`prompts/react_system_prompt.md`](prompts/react_system_prompt.md) |
| **Tool budget** | Up to **5 rounds**, **2 calls per round**, and **10 calls total** |
| **Retrieval** | Top-k: **10** by default, **20** maximum; **100** vector candidates |
| **Corpus snapshot** | The same index `build_id` |
| **Retrieval models** | The same pinned embedding and reranker revisions |

**Record every deviation.** Save the output budget, repetition penalty, reasoning effort,
and retry settings alongside each run, including any changes to the shared configuration.

The [`atlas-benchmark-eval`](.qoder/skills/atlas-benchmark-eval/SKILL.md) skill enforces this
checklist and records a configuration JSON for each run.
