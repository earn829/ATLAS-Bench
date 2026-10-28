# Tested model profiles

All profiles use chat completions, `max_top_k=20`, recommendation limit 10, GPU0 retrieval, and the shared v6 prompt. `default` below means no reasoning-control field is sent; labels such as Max-like are empirical descriptions, not explicit API settings.

| Profile | Model ID | Endpoint | Thinking control | Temp. | Output | Workers | Notes |
|---|---|---|---|---:|---:|---:|---|
| `qwen35-local-no-think` | `Qwen3.5-4B` | local `:8001/v1` | `enable_thinking=false` | 0.1 | 8192 | 16 | vLLM; `chat_template_kwargs` |
| `qwen35-local-think` | `Qwen3.5-4B` | local `:8001/v1` | `enable_thinking=true` | 0.1 | 8192 | 16 | RP 1.15; retry budget 16384 |
| `qwen36-a3b-no-think` | `Qwen3.6-35B-A3B` | local `:8001/v1` | `enable_thinking=false` | 0.1 | 8192 | 8 | vLLM; verify served ID |
| `qwen36-a3b-think` | `Qwen3.6-35B-A3B` | local `:8001/v1` | `enable_thinking=true` | 0.1 | 16384 | 8 | verify served ID |
| `qwen38-local-no-think` | `Qwen3.8-27B` | local `:8001/v1` | `enable_thinking=false` | 0.1 | 8192 | 16 | vLLM; `chat_template_kwargs` |
| `qwen38-local-think` | `Qwen3.8-27B` | local `:8001/v1` | `enable_thinking=true` | 0.1 | 8192 | 12 | RP 1.05; retry budget 16384 |
| `qwen38-max` | `qwen3.8-max` | Hosted | default/Max-like | 0.1 | 8192 | 12 | effort field ignored; top-level `enable_thinking=false` can disable |
| `deepseek-v41-flash` | `deepseek-v4.1-flash` | Hosted | forced thinking | 0.1 | 8192 | 6 | effort accepted but not reliably monotonic |
| `gemini38-flash` | `gemini-3.8-flash` | Hosted | hidden thinking | 0.1 | 8192 | 6 | uppercase `STOP` may be returned |
| `gpt56-luna` | `azure/gpt-5.6-luna` | Hosted | default Medium | omit | 8192 | 12 | `reasoning_effort=none` disables; reasoning text hidden |
| `gpt6-luna-xhigh` | `gpt-6-luna` | Hosted | `reasoning_effort=xhigh` | omit | 8192 | 12 | reasoning text hidden |
| `glm53` | `glm-5.3` | Hosted | default Max-like | 0.1 | 16384 | 12 | low/high/max supported; low for pathological retries only |
| `kimi-k3` | `kimi-k3` | Hosted | default Max-like | omit | 8192 | 12 | `none` does not disable; effort levels accepted |
| `claude-sonnet5-bedrock` | `bedrock/claude-sonnet-5` | Hosted | adaptive hidden | omit | 8192 | 1 | cache system prompt; use 8 HTTP retries |

## Local serving

Serve local Qwen models on GPU1 and reserve GPU0 for retrieval. Confirm the exact model appears at `/v1/models` before any smoke run. A known Qwen3.8-27B launch shape is:

```bash
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /data/wff_envs/q36/bin/vllm serve /path/to/Qwen3.8-27B \
  --served-model-name Qwen3.8-27B \
  --host 0.0.0.0 --port 8001 \
  --tensor-parallel-size 1 \
  --max-model-len 71680 \
  --gpu-memory-utilization 0.90 \
  --safetensors-load-strategy prefetch \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
```

Use `--safetensors-load-strategy prefetch` for large checkpoints on FUSE storage. Do not terminate an existing server until no active benchmark points to it.

## Fairness notes for published runs

- Qwen3.5-4B thinking used RP 1.15; 15 failed tasks were recovered at 16384 output tokens.
- Qwen3.8-27B thinking used RP 1.05; 8 failed tasks were recovered at 16384 output tokens.
- GLM-5.3 used 16384 for the main run; three pathological tasks were recovered with `reasoning_effort=low` and 8192.
- GPT-5.6-Luna and GPT-6-Luna require temperature omission (provider default 1).
- Kimi-K3 requires temperature omission.
- Claude uses explicit system-prompt caching and one worker because the route rate-limits aggressively.
