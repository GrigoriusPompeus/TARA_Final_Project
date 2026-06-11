"""Steered batched generation for the tilt-measurement phase.

We want to sample, for each biased prompt x':
    - n_agreeing  responses that endorse the user's stated belief (A=1)
    - n_correcting responses that state the factually correct answer (A=0)

These two distributions are obtained by *priming* the base model rather than
by using a system prompt (Llama-3.2-1B Base has no chat template). We add a
short seed continuation that biases the rest toward agreement or correction:

    AGREE_PRIME:  "Yes, "
    CORRECT_PRIME: "Actually, "

These are the same primes Sharma et al. and Shapira et al. used to elicit
"endorse" vs "correct" continuations on base models. The eventual reward
score is computed over the *whole* generated text (prime + continuation),
so the prime doesn't artificially inflate ArmoRM's per-objective verbosity
head differently across the two groups.

Why not use the user-belief or correct-answer string verbatim as the prime?
Because that would push the ArmoRM "correctness" head deterministically — we
want to measure how the reward shifts when the model has freedom to *argue
for* one stance or the other, not when we paste the answer in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch

from src.config import CFG
from src.data.few_shot import build_prefix
from src.models.policy import Policy

Group = Literal["agree", "correct"]

DEFAULT_PRIMES: dict[Group, str] = {
    "agree": "Yes, ",
    "correct": "Actually, ",
}


@dataclass
class SteerConfig:
    primes: dict[Group, str] = field(
        default_factory=lambda: dict(DEFAULT_PRIMES)
    )
    use_few_shot: bool = True
    user_tag: str = "Q: "
    assistant_tag: str = "A: "
    # natural stop pattern: next Q in the few-shot pattern
    stop_string: str = "\nQ:"


def build_steered_prompt(
    question_text: str,
    group: Group,
    cfg: SteerConfig | None = None,
) -> str:
    """Build the input string ready for tokenization."""
    cfg = cfg or SteerConfig()
    prefix = build_prefix() if cfg.use_few_shot else ""
    prime = cfg.primes[group]
    return f"{prefix}{cfg.user_tag}{question_text}\n{cfg.assistant_tag}{prime}"


@torch.no_grad()
def generate_group(
    policy: Policy,
    question_texts: list[str],
    group: Group,
    *,
    n_per_prompt: int,
    max_new_tokens: int = 96,
    temperature: float = 1.0,
    top_p: float = 0.95,
    seed: int = 42,
    batch_size: int = 8,
    steer_cfg: SteerConfig | None = None,
) -> list[list[str]]:
    """Generate n_per_prompt responses per prompt for one steering group.

    Returns: list aligned with question_texts; each entry has n_per_prompt
    continuation strings. The prime ("Yes, " / "Actually, ") is INCLUDED in
    the returned text so downstream ArmoRM scoring sees the full response.
    """
    steer_cfg = steer_cfg or SteerConfig()
    torch.manual_seed(seed)
    if policy.device == "cuda":
        torch.cuda.manual_seed_all(seed)

    full_prompts = [build_steered_prompt(q, group, steer_cfg) for q in question_texts]
    tok = policy.tokenizer
    model = policy.model

    # Replicate each prompt n_per_prompt times in a flat list.
    flat: list[tuple[int, str]] = []
    for i, p in enumerate(full_prompts):
        for _ in range(n_per_prompt):
            flat.append((i, p))

    results: list[list[str]] = [[] for _ in question_texts]
    prime = steer_cfg.primes[group]

    for start in range(0, len(flat), batch_size):
        chunk = flat[start : start + batch_size]
        texts = [t for _, t in chunk]
        idxs = [i for i, _ in chunk]
        enc = tok(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(policy.device)
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tok.pad_token_id,
        )
        gen_only = out[:, enc.input_ids.shape[1] :]
        decoded = tok.batch_decode(gen_only, skip_special_tokens=True)
        for idx, txt in zip(idxs, decoded):
            full = prime + txt
            stop = full.find(steer_cfg.stop_string)
            if stop != -1:
                full = full[:stop]
            results[idx].append(full.strip())

    return results


def generate_paired(
    policy: Policy,
    question_texts: list[str],
    *,
    n_agreeing: int | None = None,
    n_correcting: int | None = None,
    **kw,
) -> dict[Group, list[list[str]]]:
    """Generate both groups for the same set of prompts."""
    n_agreeing = n_agreeing or CFG.bon.n_agreeing_per_prompt
    n_correcting = n_correcting or CFG.bon.n_correcting_per_prompt
    return {
        "agree": generate_group(
            policy, question_texts, "agree", n_per_prompt=n_agreeing, **kw
        ),
        "correct": generate_group(
            policy, question_texts, "correct", n_per_prompt=n_correcting, **kw
        ),
    }
