#!/usr/bin/env bash
# Tool-use flexibility judge over all 14 models with DeepSeek-Pro.
set -euo pipefail

CALLER_PWD="$PWD"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  printf '%s\n' \
    'Usage: run_all_tool_flexibility.sh [smoke] [options]' \
    '  With no single-run options, judge all 14 original model runs.' \
    '  --run FILE --label LABEL --output FILE  Judge exactly one run (all required).' \
    '  --runs-root DIR             All-model inputs (default: ROOT/outputs).' \
    '  --workers N                 Worker count (default: WORKERS or 20).' \
    '  smoke, --smoke              Limit 15, default 5 workers; all-model outputs in smoke/.' \
    '  --judge-url URL             Endpoint (default: $ATLAS_JUDGE_URL); required unless --dry-run.' \
    '  --judge-model MODEL         Default: ATLAS_JUDGE_MODEL or deepseek-v4-pro.' \
    '  --judge-api-key-env NAME     Default: ATLAS_JUDGE_API_KEY; pass a name, not a key.' \
    '  --dry-run                   Forward to the judge without API calls.' \
    '  -h, --help                  Show help without loading .env.' \
    'Relative file paths and slash-containing PYTHON paths use the caller directory.' \
    'PYTHON defaults to python3. ROOT/.env is loaded only if it exists.' \
    'All-model outputs: ROOT/judge/tool_flexibility_deepseek_pro/LABEL.jsonl.'
}
fail() { printf 'Error: %s\n' "$*" >&2; exit 2; }
require_value() {
  [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || fail "missing value for $1"
}
caller_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "$CALLER_PWD" "$1" ;;
  esac
}

RUN=""; LABEL=""; OUTPUT=""; RUNS_ROOT="$ROOT/outputs"
WORKERS_OPTION=""; JUDGE_MODEL=""; JUDGE_API_KEY_ENV=""
SMOKE=0; DRY_RUN=0
while (( $# )); do
  case "$1" in
    --run|--output|--runs-root)
      require_value "$@"
      case "$1" in
        --run) RUN="$(caller_path "$2")" ;;
        --output) OUTPUT="$(caller_path "$2")" ;;
        --runs-root) RUNS_ROOT="$(caller_path "$2")" ;;
      esac
      shift ;;
    --label|--workers|--judge-url|--judge-model|--judge-api-key-env)
      require_value "$@"
      case "$1" in
        --label) LABEL="$2" ;;
        --workers) WORKERS_OPTION="$2" ;;
        --judge-url) JUDGE_URL="$2" ;;
        --judge-model) JUDGE_MODEL="$2" ;;
        --judge-api-key-env) JUDGE_API_KEY_ENV="$2" ;;
      esac
      shift ;;
    smoke|--smoke) SMOKE=1 ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help) usage; exit 0 ;;
    *) fail "unknown option: $1 (see --help)" ;;
  esac
  shift
done

OUT_DIR="$ROOT/judge/tool_flexibility_deepseek_pro"
if (( SMOKE )); then OUT_DIR="$OUT_DIR/smoke"; fi
LABELS=(); RUN_FILES=(); OUTPUT_FILES=()
select_run() {
  LABELS+=("$1")
  RUN_FILES+=("$RUNS_ROOT/$2/subset_1200.final.jsonl")
  OUTPUT_FILES+=("$OUT_DIR/$1.jsonl")
}
SINGLE_MODE=0
if [[ -n "$RUN" || -n "$LABEL" || -n "$OUTPUT" ]]; then
  [[ -n "$RUN" && -n "$LABEL" && -n "$OUTPUT" ]] || fail '--run, --label and --output must be supplied together'
  SINGLE_MODE=1
  LABELS=("$LABEL"); RUN_FILES=("$RUN"); OUTPUT_FILES=("$OUTPUT")
else
  select_run qwen35_4b_nothink qwen3.5-4B-local-nothinking-topk20-reclimit10
  select_run qwen35_4b_think qwen3.5-4B-local-thinking-rp115-topk20-reclimit10
  select_run qwen36_a3b_nothink qwen3.6-35B-A3B-nothinking-topk20-reclimit10
  select_run qwen36_a3b_think qwen3.6-35B-A3B-thinking-topk20-reclimit10
  select_run qwen38_27b_nothink qwen3.8-27B-local-nothinking-topk20-reclimit10
  select_run qwen38_27b_think qwen3.8-27B-local-thinking-rp105-topk20-reclimit10
  select_run qwen38_max qwen3.8-max-hosted-default-thinking-topk20-reclimit10
  select_run deepseek_v41_flash deepseek-v4-1-flash-hosted-topk20-reclimit10
  select_run claude5_sonnet bedrock-claude-sonnet-5-hosted-default-thinking-cache-topk20-reclimit10
  select_run gpt56_luna azure-gpt-5-6-luna-hosted-default-thinking-topk20-reclimit10
  select_run gpt6_luna gpt-6-luna-hosted-xhigh-topk20-reclimit10
  select_run glm53 glm-5-3-hosted-default-thinking-topk20-reclimit10
  select_run kimi_k3 kimi-k3-hosted-default-thinking-topk20-reclimit10
  select_run gemini38_flash gemini-3-8-flash-hosted-default-thinking-topk20-reclimit10
fi

# Preflight every selected input before invoking a judge or creating output directories.
for i in "${!LABELS[@]}"; do
  label="${LABELS[$i]}"
  [[ -n "$label" && "$label" != . && "$label" != .. && "$label" != */* && "$label" != *\\* ]] || fail "unsafe label: $label"
  [[ -f "${RUN_FILES[$i]}" ]] || fail "run file does not exist: ${RUN_FILES[$i]}"
done

if [[ -f "$ROOT/.env" ]]; then
  set -a; source "$ROOT/.env"; set +a
fi
PYTHON="${PYTHON:-python3}"
if [[ "$PYTHON" == */* ]]; then PYTHON="$(caller_path "$PYTHON")"; fi
WORKERS="${WORKERS:-20}"
if (( SMOKE )); then WORKERS=5; fi
WORKERS="${WORKERS_OPTION:-$WORKERS}"
[[ "$WORKERS" =~ ^[1-9][0-9]*$ ]] || fail '--workers/WORKERS must be a positive integer'
JUDGE_URL="${JUDGE_URL:-${ATLAS_JUDGE_URL:-}}"
JUDGE_MODEL="${JUDGE_MODEL:-${ATLAS_JUDGE_MODEL:-deepseek-v4-pro}}"
JUDGE_API_KEY_ENV="${JUDGE_API_KEY_ENV:-ATLAS_JUDGE_API_KEY}"
JUDGE_ARGS=(--workers "$WORKERS" --judge-model "$JUDGE_MODEL" --judge-api-key-env "$JUDGE_API_KEY_ENV")
if [[ -n "$JUDGE_URL" ]]; then
  JUDGE_ARGS+=(--judge-url "$JUDGE_URL")
elif (( ! DRY_RUN )); then
  fail '--judge-url or ATLAS_JUDGE_URL is required (except with --dry-run)'
fi
if (( SMOKE )); then JUDGE_ARGS+=(--limit 15); fi
if (( DRY_RUN )); then JUDGE_ARGS+=(--dry-run); fi
cd -- "$ROOT"

for i in "${!LABELS[@]}"; do
  printf '=== START %s ===\n' "${LABELS[$i]}"
  "$PYTHON" "$ROOT/judge/tool_flexibility_judge.py" \
    --run "${RUN_FILES[$i]}" --label "${LABELS[$i]}" --output "${OUTPUT_FILES[$i]}" \
    "${JUDGE_ARGS[@]}"
  printf '=== DONE %s ===\n' "${LABELS[$i]}"
done
if (( ! SINGLE_MODE )); then printf '%s\n' '=== ALL 14 TOOL-FLEXIBILITY JUDGES COMPLETE ==='; fi
