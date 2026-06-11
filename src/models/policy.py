"""Llama-3.2-1B Base policy loader and batched generation helper.

Keeps the base model frozen (no SFT, no RLHF) — Shapira's covariance theorem
requires the base policy be untouched. Coherent generation is induced via a
few-shot prefix in the input, not via weight changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import CFG


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device: str) -> torch.dtype:
    if device == "cuda":
        return torch.bfloat16
    if device == "mps":
        # MPS supports bf16 on macOS 14+; fall back to fp16 on older
        return torch.bfloat16
    return torch.float32


@dataclass
class Policy:
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    device: str
    dtype: torch.dtype


def load_policy(
    model_id: str | None = None,
    device: str | None = None,
    dtype: torch.dtype | None = None,
) -> Policy:
    model_id = model_id or CFG.models.policy
    device = device or pick_device()
    dtype = dtype or pick_dtype(device)

    tok = AutoTokenizer.from_pretrained(model_id, token=os.environ.get("HF_TOKEN"))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # IMPORTANT: pad LEFT for generation, otherwise the generated tokens get
    # prepended to padding instead of continuing the prompt.
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        token=os.environ.get("HF_TOKEN"),
    )
    model.to(device).eval()
    return Policy(model=model, tokenizer=tok, device=device, dtype=dtype)


@torch.no_grad()
def batched_generate(
    policy: Policy,
    prompts: list[str],
    *,
    n_per_prompt: int = 1,
    max_new_tokens: int = 96,
    temperature: float = 1.0,
    top_p: float = 1.0,
    seed: int | None = None,
    batch_size: int = 8,
) -> list[list[str]]:
    """Generate `n_per_prompt` continuations for each prompt.

    Returns a list aligned with `prompts`, each entry a list of n continuations
    (decoded with the prompt itself stripped).
    """
    if seed is not None:
        torch.manual_seed(seed)
        if policy.device == "cuda":
            torch.cuda.manual_seed_all(seed)

    do_sample = temperature > 0 and n_per_prompt > 1
    results: list[list[str]] = [[] for _ in prompts]

    # Repeat each prompt n times and batch over (prompt × sample).
    flat_prompts: list[tuple[int, str]] = []
    for i, p in enumerate(prompts):
        for _ in range(n_per_prompt):
            flat_prompts.append((i, p))

    tok = policy.tokenizer
    model = policy.model

    for start in range(0, len(flat_prompts), batch_size):
        batch = flat_prompts[start : start + batch_size]
        texts = [t for _, t in batch]
        idxs = [i for i, _ in batch]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(
            policy.device
        )
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            top_p=top_p,
            pad_token_id=tok.pad_token_id,
        )
        # strip the input portion from each output
        gen_only = out[:, enc.input_ids.shape[1] :]
        decoded = tok.batch_decode(gen_only, skip_special_tokens=True)
        for idx, txt in zip(idxs, decoded):
            results[idx].append(txt.strip())

    return results


@torch.no_grad()
def logprob_of_completion(
    policy: Policy,
    prompt: str,
    completion: str,
) -> float:
    """Sum log P(completion_tokens | prompt) under the policy.

    Used for MC1/MC2 evaluation: score each candidate answer by its log-prob.
    """
    tok = policy.tokenizer
    model = policy.model

    full = prompt + completion
    enc_full = tok(full, return_tensors="pt", truncation=True, max_length=2048).to(policy.device)
    enc_prompt = tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to(policy.device)
    prompt_len = enc_prompt.input_ids.shape[1]

    out = model(**enc_full)
    logits = out.logits[0, :-1, :]  # next-token prediction at each position
    targets = enc_full.input_ids[0, 1:]  # shifted targets
    logp_all = torch.log_softmax(logits.float(), dim=-1)
    tgt_logp = logp_all.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    # only sum log-probs for tokens that belong to the completion
    completion_logp = tgt_logp[prompt_len - 1 :]
    return float(completion_logp.sum().item())


@torch.no_grad()
def batched_logprob_of_completions(
    policy: Policy,
    prompt: str,
    completions: list[str],
) -> list[float]:
    """Vectorized version of logprob_of_completion for a fixed prompt."""
    return [logprob_of_completion(policy, prompt, c) for c in completions]
