"""ArmoRM-Llama3-8B reward model loader.

Wang et al. (EMNLP 2024) arXiv:2406.12845. Absolute-Rating Multi-Objective
Reward Model with a Mixture-of-Experts gating network over 19 interpretable
heads (helpfulness, correctness, honesty, verbosity, safety, ...).

Two modes:
    bf16  — cloud A100. Full precision, accurate scoring, ~16 GB.
    8bit  — local M1 Max via bitsandbytes... not supported on Mac!
            On Mac we'll have to do bf16 too (fits in 32 GB but tight) or
            quantize via mlx_lm. For the initial Phase-2 pilot we will run
            on the cloud anyway because local generation + scoring of
            128 candidates × 1000 prompts is impractical on M1.

We expose two scoring outputs:
    score:   the scalar gated-MoE output (the standard ArmoRM "preference" score)
    heads:   the 19 per-objective head scores (for style-vs-factual decomposition)

Reference: ArmoRM source at huggingface.co/RLHFlow/ArmoRM-Llama3-8B-v0.1
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence

import torch
import transformers.models.llama.modeling_llama as _llama_mod
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import CFG

# Compatibility shim: ArmoRM's modeling_custom.py imports `LLAMA_INPUTS_DOCSTRING`
# from transformers.models.llama.modeling_llama. That constant was removed
# circa transformers 4.45 as part of a docstring refactor. It's only used as
# the argument to an `add_start_docstrings_to_model_forward(...)` decorator,
# so an empty string is functionally equivalent. We inject it BEFORE the
# `trust_remote_code` import path imports the file.
if not hasattr(_llama_mod, "LLAMA_INPUTS_DOCSTRING"):
    _llama_mod.LLAMA_INPUTS_DOCSTRING = ""

# Names of the 19 ArmoRM objectives, in head order.
# Source: model card on HF. Kept here so analysis can label heads without
# requiring the model object at plot time.
ARMORM_HEAD_NAMES: tuple[str, ...] = (
    "helpsteer-helpfulness",
    "helpsteer-correctness",
    "helpsteer-coherence",
    "helpsteer-complexity",
    "helpsteer-verbosity",
    "ultrafeedback-overall_score",
    "ultrafeedback-instruction_following",
    "ultrafeedback-truthfulness",
    "ultrafeedback-honesty",
    "ultrafeedback-helpfulness",
    "beavertails-is_safe",
    "prometheus-score",
    "argilla-overall_quality",
    "argilla-judge_lm",
    "code-complexity",
    "code-style",
    "code-explanation",
    "code-instruction-following",
    "code-readability",
)


@dataclass
class Reward:
    model: AutoModelForSequenceClassification
    tokenizer: AutoTokenizer
    device: str
    dtype: torch.dtype


def load_reward(
    model_id: str | None = None,
    device: str | None = None,
    dtype: torch.dtype | None = None,
) -> Reward:
    model_id = model_id or CFG.models.reward
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    if dtype is None:
        dtype = torch.bfloat16 if device in {"cuda", "mps"} else torch.float32

    tok = AutoTokenizer.from_pretrained(
        model_id, use_fast=True, token=os.environ.get("HF_TOKEN")
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    model.to(device).eval()
    return Reward(model=model, tokenizer=tok, device=device, dtype=dtype)


def _format_pair(tokenizer, prompt: str, response: str) -> str:
    """Apply the ArmoRM chat template (user/assistant turns) to a (prompt, response)."""
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


@torch.no_grad()
def score(
    reward: Reward,
    prompt: str,
    responses: Sequence[str],
    *,
    batch_size: int = 4,
    return_heads: bool = False,
) -> dict[str, list[float] | list[list[float]]]:
    """Score (prompt, response) pairs with ArmoRM.

    Returns:
        {"score": [scalar per response],
         "heads": [list[19 floats] per response]    # only if return_heads}

    The gated MoE score is exposed via the model's `score` output;
    the per-objective heads are in the auxiliary output of ArmoRM.
    """
    texts = [_format_pair(reward.tokenizer, prompt, r) for r in responses]
    scores: list[float] = []
    heads_out: list[list[float]] = []

    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        enc = reward.tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to(reward.device)
        out = reward.model(**enc)

        # ArmoRM's model card: `out.score` is the gated scalar; `out.rewards`
        # is the (B, 19) per-objective tensor. Older revisions may name these
        # differently — fall back to attribute introspection.
        gated = getattr(out, "score", None)
        if gated is None:
            gated = getattr(out, "logits", None)
        per_head = getattr(out, "rewards", None)

        if gated is not None:
            scores.extend(gated.flatten().float().cpu().tolist())
        if return_heads and per_head is not None:
            heads_out.extend(per_head.float().cpu().tolist())

    result: dict[str, list[float] | list[list[float]]] = {"score": scores}
    if return_heads:
        result["heads"] = heads_out
    return result
