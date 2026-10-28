# Data

ATLAS-Bench is built on a real-world local-life corpus covering three districts:
Chaoyang (Beijing), Huangpu (Shanghai), and Dabancheng (Urumqi). It contains
**425,244 POIs** and **260,757 user notes**, with 1,200 evaluation queries
(see `../query/`).

## Directory layout

```
data/
├── indexes/    Retrieval indexes shipped with the repo (Git LFS)
├── raw/        Sanitized source JSON the indexes were built from (Git LFS)
└── models/     Embedding & reranker weights (NOT shipped; download below)
```

### `indexes/`

| File | Contents |
|---|---|
| `poi_rag.sqlite3` | POI records, full-text (FTS5) index, spatial (RTree) index |
| `note_rag.sqlite3` | Note records, full-text index, per-POI lookup index |
| `poi_vectors.faiss`, `note_vectors.faiss` | Qwen3-Embedding-0.6B dense vector indexes |
| `manifest.json`, `vector_manifest.json` | Build metadata and a shared `build_id` that ties lexical and vector indexes together |

### `raw/`

The three source JSON files (POI features with embedded user notes) used to
build the indexes, released in sanitized form: the structure and all content
fields (names, coordinates, addresses, tags, note text) are unchanged, and
only the internal POI / note identifiers are replaced with sequential
`P######` / `N######` ids. The same mapping is applied consistently across
the raw files, both indexes, and the query set, so cross-references remain
intact.

### Anonymized identifiers

All POI and note ids in this repository are anonymized sequential ids. They
are internally consistent everywhere (a note's `poiid` always resolves to the
same POI), but they cannot be mapped back to the original platform's
identifiers.

## Embedding & reranker weights

Model weights (~2.3 GB) are not shipped. Before the first run:

```bash
pip install -r requirements-vector.txt
python3 scripts/fetch_models.py          # download to data/models/
python3 scripts/fetch_models.py --check  # verify pinned revisions
```

Pinned revisions (must match — the shipped FAISS indexes were built with
exactly these checkpoints; other commits silently degrade retrieval):

- `Qwen/Qwen3-Embedding-0.6B` @ `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`
- `Qwen/Qwen3-Reranker-0.6B` @ `e61197ed45024b0ed8a2d74b80b4d909f1255473`

Set `HF_ENDPOINT` to use a mirror, or `--cache-dir` (with
`LOCAL_LIFE_VECTOR_CACHE`) to choose another download location.

## Rebuilding the indexes

To rebuild from the released raw data:

```bash
python3 scripts/build_indexes.py \
  --source data/raw/chaoyang_poi_features.json \
           data/raw/huangpu_poi_features.json \
           data/raw/dabancheng_poi_features.json
python3 scripts/build_vector_indexes.py   # requires requirements-vector.txt + models
```

Add `--overwrite` to `build_indexes.py` when the output directory already
contains an index. Rebuilt indexes are validated against the shipped ones by
record counts, schema version, and `build_id`; mismatched files are rejected
at load time rather than silently substituted.
