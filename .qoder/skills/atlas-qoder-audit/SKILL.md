---
name: atlas-qoder-audit
description: Run the ATLAS-Bench Qoder blind constraint audit that produces the CSR, CR and Success@k answer metrics. Use when adjudicating a finalized run's recommendations against judge/RUBRIC.md, rebuilding blind audit inputs, checking audit progress or pending chunks, validating verdict JSONL, or scoring verdicts with the published answer-metric formulas.
---

# ATLAS Qoder Blind Audit

Reproduce the recommendation-level constraint audit behind CSR, CR and Success@k: build blind
per-query payloads from a finalized run, adjudicate every recommendation against `judge/RUBRIC.md`,
then validate and score the verdicts with the published formulas.

Run every command from the repository root. Use `${PYTHON:-python3}`. Nothing here calls a model API
or needs credentials; the adjudicator is the Qoder session itself.

## Equivalence invariants

Breaking any of these breaks comparability with the published table.

- `judge/RUBRIC.md` (V3) is the single source of truth. Read it in full before adjudicating. Never
  paraphrase, summarize, extend or patch it. A rubric edit ends the campaign: start a new label.
- Stay blind. Judge only the payload fields. Never open `record.answer` in `query/subset_1200.json`,
  gold or click-target POIs, `outputs/**/*.jsonl`, or anything else that reveals the reference answer
  or model identity — and never ask or accept which model produced a chunk.
- One chunk is one independent session. No cross-chunk, cross-model or cross-replicate calibration;
  do not read earlier verdicts to anchor a new one.
- Inputs must come from `judge/build_qoder_judge_inputs.py` with its default 60000-character
  budget. Hand-edited or re-chunked inputs change what the auditor sees.
- Exactly one verdict row per recommendation, in input order. Queries with an empty answer get zero
  rows — that is a valid, scored outcome, not a gap.
- Published numbers come only from `scripts/score_verdicts.py` in this skill.
  `judge/summarize_qoder_judge.py` is the legacy qualified-only view; it does not match the paper.
- No selective re-judging. A replicate covers every arm uniformly, and both replicates get reported.
  Measured test–retest swings reach ~5.8 pp, so treat gaps under ~2 pp as noise, not a result.

## Workflow

### 1. Preflight

Confirm the run is finalized (`subset_1200.final.jsonl` with `records=1200`, `missing=0`; see the
`atlas-benchmark-eval` skill) and that `judge/RUBRIC.md` is present and unmodified. Pick a label: one
safe path component `[A-Za-z0-9][A-Za-z0-9_.-]*`, e.g. `qwen38_27b_think`, or `qwen38_27b_think_r2`
for a replicate. Never reuse a label whose verdicts came from a different rubric or run.

### 2. Build blind inputs

```bash
python3 judge/build_qoder_judge_inputs.py \
    --run outputs/<rundir>/subset_1200.final.jsonl --label <label> --judge-dir judge
```

Writes `judge/<label>/inputs/chunk_NN.json` (query, context, annotated constraints, and each
recommendation's `cited_evidence` + `observed_facts`) plus an empty `judge/<label>/verdicts/`, and
prints the query, recommendation and chunk counts. Optional: `--tasks` for a non-default annotation
file, `--budget-chars` only when deliberately reproducing a legacy chunking. Record the printed
counts; step 4 checks verdicts against them.

### 3. Adjudicate chunk by chunk

Get the worklist:

```bash
python3 .qoder/skills/atlas-qoder-audit/scripts/audit_status.py --judge-dir judge --pending <label>
```

For each pending `chunk_NN`, in order:

1. Read `judge/RUBRIC.md`, then `judge/<label>/inputs/chunk_NN.json`.
2. Decide each recommendation independently under the rubric's `qualified` / `acceptable` /
   `unacceptable` definitions and its 13 rules, using only that recommendation's own
   `cited_evidence` and `observed_facts`.
3. Append one JSON line per recommendation to `judge/<label>/verdicts/chunk_NN.jsonl` — same `NN` as
   the input, query order then ascending `rank`, with `task_id`, `rank`, `poiid`, `verdict`,
   `constraints_met`, `reason`. Field semantics and the hard validation rules are in
   `references/verdict-contract.md`.
4. Re-run `audit_status.py <label>` and clear every anomaly before opening the next chunk.

Keep one chunk in context at a time. Do not merge chunks into a single write, and do not back-fill a
finished chunk from memory of another one.

### 4. Validate and score

```bash
python3 .qoder/skills/atlas-qoder-audit/scripts/score_verdicts.py --judge-dir judge <label>
```

Validates the contract first and refuses to score on any violation (exit 2, offending rows listed).
On success it writes `judge/<label>/summary_published.json` and `judge/_published_metrics.md`, and
prints CSR, CR, Success@k, Recs/query, coverage and the verdict distribution. Add `--no-write` for a
read-only pass, `--json` for machine output, or several labels to compare arms in one table.
Coverage below 1.000 means unfinished adjudication: the metrics then cover only fully judged
queries, so never report them as final.

### 5. Report

Report CSR, CR, Success@k, Recs/query with `coverage=1.000`, the query and should-answer counts,
verdict counts, and every deviation (rubric version, label, budget, replicate). For arm comparisons,
compare identical `task_id`s and report verdict transitions, not only aggregates. State explicitly
that these are LLM-auditor verdicts with measured test–retest noise.

## Resources

- `references/verdict-contract.md`: verdict row schema, ordering, validation rules, and the published
  CSR / CR / Success@k / Recs-per-query formulas with their denominators.
- `scripts/audit_status.py`: read-only per-chunk progress, pending worklist, anomaly detection.
- `scripts/score_verdicts.py`: contract validation plus published-metric scoring.
- `judge/RUBRIC.md`: the adjudication rules themselves (owned by the repository, not this skill).
- `judge/build_qoder_judge_inputs.py`: blind input builder.
- `.qoder/skills/atlas-benchmark-eval/`: producing the finalized run that this skill audits.
