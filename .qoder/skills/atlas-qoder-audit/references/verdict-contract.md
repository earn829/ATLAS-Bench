# Verdict contract and published metric formulas

## Blind input payload

`judge/<label>/inputs/chunk_NN.json` is a JSON array; one element per query, in annotation order:

| Field | Meaning |
|---|---|
| `task_id` | Query identifier, e.g. `v10_0421`. |
| `query` | User query text. |
| `context` | Query context (`time`, `loc`, ...). |
| `constraints` | Annotated constraint list: `cid`, `text`, `h` (刚性/柔性), `s`, `evidence_type`, `expected_predicate`. |
| `n_constraints` | Length of `constraints`; the denominator for `constraints_met`. |
| `satisfiable` | `false` marks an unanswerable query; anything else is a should-answer query. |
| `recommendations` | Ranked recommendations, each with `rank`, `poiid`, `name`, `reason`, `constraint_gaps`, `evidence_ids`, `cited_evidence`, `observed_facts`. |

Reference answers, gold and click-target POIs, and model identity are excluded by construction. Do not
recover them from any other file.

## Verdict row

`judge/<label>/verdicts/chunk_NN.jsonl` — one JSON object per line, one line per recommendation:

```json
{"task_id":"v10_0421","rank":1,"poiid":"B0MUYSP2GI","verdict":"qualified","constraints_met":1,"reason":"简短依据"}
```

| Field | Rule |
|---|---|
| `task_id` | Must be a query from the same-numbered input chunk. |
| `rank` | 1-based, matching the input recommendation at that rank. |
| `poiid` | Must equal the input `poiid` for that rank; never re-typed from memory. |
| `verdict` | Exactly one of `qualified`, `acceptable`, `unacceptable` per `judge/RUBRIC.md`. |
| `constraints_met` | Integer in `0..n_constraints`: how many annotated constraints (rigid **and** soft) are fully satisfied by that POI's own evidence. `qualified` requires `constraints_met == n_constraints`. `acceptable` and `unacceptable` must state the honest count; a soft constraint only disclosed in `constraint_gaps` without positive evidence counts 0. |
| `reason` | Non-empty short justification citing the deciding evidence. |

Ordering: queries in input order, and within a query ascending `rank`. Extra fields are ignored by
the scorer; missing or malformed required fields are contract violations.

Coverage rule: a query with `k > 0` recommendations needs exactly `k` rows; a query with an empty
recommendation list needs zero rows and still counts toward every metric.

## Validation enforced by `scripts/score_verdicts.py`

Any of these aborts scoring with exit 2 and prints up to 20 offending rows:

1. `task_id` absent from the blind inputs.
2. A row written to a chunk other than the one containing its query.
3. Duplicate `(task_id, rank)`.
4. `rank` not among the input recommendations, or `poiid` mismatching that rank.
5. `verdict` outside the allowed set.
6. `constraints_met` not an integer, outside `0..n_constraints`, or `< n_constraints` on a
   `qualified` row.
7. Empty or non-string `reason`.
8. Duplicate `task_id` across input chunks.

Queries with fewer rows than recommendations are excluded from the metrics and reported through
`coverage`, never silently treated as failures. `coverage = judged_rows / expected_rows`, where
`judged_rows` counts only rows belonging to fully judged queries and `expected_rows` counts every
recommendation in the blind inputs; a finished campaign must show `coverage = 1.0`. `written_rows`
reports how many rows exist on disk, so a partial chunk is visible even when nothing is scorable yet.

## Chunk states reported by `scripts/audit_status.py`

| State | Meaning |
|---|---|
| `missing` | No `verdicts/chunk_NN.jsonl` yet. |
| `incomplete` | The file exists but some query has fewer rows than recommendations. |
| `invalid` | Row count is right but at least one row breaks the contract (bad verdict, wrong `poiid`, duplicate, `constraints_met` out of range, or `qualified` with `constraints_met < n_constraints`). |
| `complete` | Every query has exactly its rows and no anomaly was found. |

A label is `COMPLETE` only when all its chunks are `complete`. `--pending` lists every chunk that is
not `complete`, so an invalid chunk stays on the worklist until it is rewritten.

## Published formulas

`GOOD = {qualified, acceptable}`. Let `R_i` be the judged rows of query `i`, `n_i` its
`n_constraints`, and `S = {i : satisfiable_i is not false}` the should-answer subset. All averages are
micro over queries — **not** macro over cities; the per-city view is a separate artifact produced by
`judge/analyze_regional_results.py`.

```
den        = Σ_i |R_i| · n_i
CSR        = 100 · Σ_i Σ_{r ∈ R_i, verdict(r) ∈ GOOD} constraints_met(r) / den
CR         = 100 · Σ_i Σ_{r ∈ R_i}                    constraints_met(r) / den
Success@k  = 100 · mean_{i ∈ S} 1[∃ r ∈ R_i : verdict(r) ∈ GOOD]
Recs/query =        mean_{i ∈ S} |recommendations_i|
```

CSR is the strict rate: only constraints met by recommendations that are themselves `qualified` or
`acceptable` count. CR is the lenient rate over every judged recommendation. Success@k asks whether at
least one good recommendation exists, and an empty answer on a should-answer query scores 0.

On the full 1200-query set `|S| = 1004`. These formulas reproduce all 14 published rows, e.g.
`qwen38_27b_think` CSR 61.24 / CR 77.92 / Success@k 82.87 / Recs 2.98, and `qwen35_4b_nothink`
37.34 / 65.83 / 53.98 / 2.24.

## Not the published definition

`judge/summarize_qoder_judge.py` reports qualified-only ratios and a Success@k over non-empty
queries only. It is kept for audit inspection; its numbers are systematically lower than the table and
must not be quoted as CSR, CR or Success@k.
