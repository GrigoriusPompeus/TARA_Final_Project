"""Reward functions for GRPO.

We expose two:
    armorm_reward(prompts, completions, **info) -> list[float]
        Pure ArmoRM gated MoE score. Used for the *unmitigated* baseline.
    corrected_reward(prompts, completions, **info) -> list[float]
        Shapira Theorem 6 corrected reward:
            r_corr(x, y) = r(x, y) − λ · A(x, y) · 1{x ∈ X_false}
        Uses the agreement detector to mark sycophantic completions on
        belief_wrong prompts only. Used for the mitigation experiment.

The TRL GRPOTrainer passes `prompts` and `completions` as lists of strings
in matching order, plus all dataset columns (including `is_false`,
`user_belief`, `correct_answer`) as keyword args. We rely on this to
keep the reward function purely a function of (x, y) without global state.

A single ArmoRM model is loaded once at module-import time and re-used
across calls. This is acceptable for a single-process training run.
"""

from __future__ import annotations

from typing import Sequence

import torch

from src.bon.agreement import agreement_indicator
from src.models.reward import Reward, load_reward, score as score_pairs


class _RewardSingleton:
    instance: Reward | None = None


def _get_reward() -> Reward:
    if _RewardSingleton.instance is None:
        _RewardSingleton.instance = load_reward()
    return _RewardSingleton.instance


def _strip_few_shot(prompt: str) -> str:
    """Remove the few-shot prefix from a prompt so ArmoRM sees only the
    real user query.

    The few-shot prefix ends with the last 'A: <answer>\n\n' before the
    final 'Q: <question>\nA: ' the model is completing. We split on the
    last 'Q: ' marker and keep what follows.
    """
    if "Q: " not in prompt:
        return prompt.strip()
    tail = prompt.rsplit("Q: ", 1)[-1]
    if tail.endswith("A: "):
        tail = tail[:-3]
    return tail.strip()


@torch.no_grad()
def armorm_reward(
    prompts: Sequence[str],
    completions: Sequence[str],
    **info,
) -> list[float]:
    """Score each (prompt, completion) pair with ArmoRM's gated MoE output."""
    reward = _get_reward()
    rewards: list[float] = []
    for p, c in zip(prompts, completions):
        clean_prompt = _strip_few_shot(p)
        r = score_pairs(reward, clean_prompt, [c], batch_size=1)
        rewards.append(r["score"][0])
    return rewards


@torch.no_grad()
def corrected_reward(
    prompts: Sequence[str],
    completions: Sequence[str],
    *,
    lam: float = 1.0,
    agreement_mode: str = "key_terms",
    overlap_threshold: float = 0.6,
    **info,
) -> list[float]:
    """Shapira Theorem 6 corrected reward.

    r_corr(x, y) = r(x, y) - lam * A(x, y) * 1{x ∈ X_false}

    `info` must contain per-example arrays:
        is_false:        list[int]
        user_belief:     list[str]
        correct_answer:  list[str]
    """
    is_false = info.get("is_false") or [0] * len(prompts)
    user_belief = info.get("user_belief") or [""] * len(prompts)
    correct_answer = info.get("correct_answer") or [""] * len(prompts)

    base = armorm_reward(prompts, completions)
    out: list[float] = []
    for p, c, b, u, ca, isf in zip(prompts, completions, base, user_belief, correct_answer, is_false):
        if isf and u:
            A = agreement_indicator(
                c, u, ca, mode=agreement_mode, overlap_threshold=overlap_threshold
            )
            out.append(b - lam * A)
        else:
            out.append(b)
    return out


def make_corrected_reward(
    *,
    lam: float = 1.0,
    agreement_mode: str = "key_terms",
    overlap_threshold: float = 0.6,
):
    """Return a (prompts, completions) -> list[float] callable that closes over
    the mitigation hyperparameters. Suitable to pass to TRL GRPOTrainer
    `reward_funcs`."""

    def _fn(prompts, completions, **info):
        return corrected_reward(
            prompts,
            completions,
            lam=lam,
            agreement_mode=agreement_mode,
            overlap_threshold=overlap_threshold,
            **info,
        )

    _fn.__name__ = f"corrected_lam{lam}"
    return _fn
