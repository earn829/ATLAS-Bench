from __future__ import annotations

import threading
from pathlib import Path
from typing import List, Sequence

from .dense_retrieval import VectorIndexError, resolve_model_source


DEFAULT_RERANK_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)


class SharedQwen3Reranker:
    """Official Qwen3 generative yes/no reranker, loaded lazily and shared."""

    def __init__(
        self,
        *,
        model_name: str,
        max_length: int,
        batch_size: int,
        cache_dir: Path,
        device: str = "auto",
        local_files_only: bool = False,
        instruction: str = DEFAULT_RERANK_INSTRUCTION,
    ) -> None:
        self.model_name = model_name
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.cache_dir = Path(cache_dir)
        self.requested_device = device
        self.local_files_only = local_files_only
        self.instruction = instruction
        self._model = None
        self._tokenizer = None
        self._device = ""
        self._prefix_tokens: List[int] = []
        self._suffix_tokens: List[int] = []
        self._true_token_id = -1
        self._false_token_id = -1
        self._lock = threading.Lock()

    @property
    def device(self) -> str:
        self._load()
        return self._device

    def _resolve_device(self) -> str:
        if self.requested_device != "auto":
            return self.requested_device
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    def _load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as exc:
                raise VectorIndexError(
                    "缺少transformers或torch，无法加载Qwen3 reranker"
                ) from exc
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._device = self._resolve_device()
            model_source = resolve_model_source(
                self.model_name,
                self.cache_dir,
                local_files_only=self.local_files_only,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_source,
                cache_dir=str(self.cache_dir),
                local_files_only=self.local_files_only,
                padding_side="left",
            )
            tokenizer.padding_side = "right" if self._device == "mps" else "left"
            dtype = (
                torch.float16
                if self._device in {"cuda", "mps"}
                else torch.float32
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_source,
                cache_dir=str(self.cache_dir),
                local_files_only=self.local_files_only,
                dtype=dtype,
            ).eval()
            model.to(self._device)
            self._true_token_id = int(tokenizer.convert_tokens_to_ids("yes"))
            self._false_token_id = int(tokenizer.convert_tokens_to_ids("no"))
            if self._true_token_id < 0 or self._false_token_id < 0:
                raise VectorIndexError("Qwen3 reranker tokenizer缺少yes/no token")
            prefix = (
                '<|im_start|>system\nJudge whether the Document meets the requirements '
                'based on the Query and the Instruct provided. Note that the answer can '
                'only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
            )
            suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            self._prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
            self._suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
            if self.max_length <= len(self._prefix_tokens) + len(self._suffix_tokens):
                raise VectorIndexError("reranker max_length不足以容纳提示词模板")
            self._tokenizer = tokenizer
            self._model = model

    def _formatted_pair(self, query: str, document: str) -> str:
        return (
            "<Instruct>: %s\n<Query>: %s\n<Document>: %s"
            % (self.instruction, query, document)
        )

    def score(self, query: str, documents: Sequence[str]) -> List[float]:
        if not documents:
            return []
        self._load()
        import torch

        scores: List[float] = []
        payload_length = self.max_length - len(self._prefix_tokens) - len(
            self._suffix_tokens
        )
        with self._lock, torch.inference_mode():
            for start in range(0, len(documents), self.batch_size):
                batch_documents = documents[start : start + self.batch_size]
                formatted = [
                    self._formatted_pair(query, str(document))
                    for document in batch_documents
                ]
                tokenized = self._tokenizer(
                    formatted,
                    padding=False,
                    truncation=True,
                    max_length=payload_length,
                    return_attention_mask=False,
                )
                token_lists = [
                    self._prefix_tokens + ids + self._suffix_tokens
                    for ids in tokenized["input_ids"]
                ]
                inputs = self._tokenizer.pad(
                    {"input_ids": token_lists},
                    padding=True,
                    return_attention_mask=True,
                    return_tensors="pt",
                )
                inputs = {key: value.to(self._device) for key, value in inputs.items()}
                model_outputs = self._model.model(
                    **inputs,
                    use_cache=False,
                    return_dict=True,
                )
                hidden_states = model_outputs.last_hidden_state
                if self._tokenizer.padding_side == "right":
                    final_positions = inputs["attention_mask"].sum(dim=1) - 1
                    batch_indices = torch.arange(
                        hidden_states.shape[0], device=hidden_states.device
                    )
                    final_hidden = hidden_states[batch_indices, final_positions]
                else:
                    final_hidden = hidden_states[:, -1, :]
                logits = self._model.lm_head(final_hidden)
                binary_logits = torch.stack(
                    (
                        logits[:, self._false_token_id],
                        logits[:, self._true_token_id],
                    ),
                    dim=1,
                )
                logit_differences = binary_logits[:, 1].float() - binary_logits[:, 0].float()
                if not torch.isfinite(logit_differences).all():
                    raise VectorIndexError(
                        "reranker生成NaN或Inf；请检查设备精度和batch size"
                    )
                scores.extend(
                    float(value) for value in logit_differences.cpu().tolist()
                )
        return scores
