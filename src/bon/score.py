"""Batched ArmoRM scoring of (prompt, response) pairs.

A thin wrapper over `src.models.reward.score` that knows how to:
    - take a list of prompts with per-prompt response groups
    - flatten to (prompt, response) pairs
    - score in batches
    - return per-prompt arrays of scores
"""

from __future__ import annotations

from typing import Sequence

from src.models.reward import Reward, score as _score


def score_per_prompt_groups(
    reward: Reward,
    prompt_texts: Sequence[str],
    response_groups: Sequence[list[str]],
    *,
    return_heads: bool = False,
    batch_size: int = 4,
) -> list[dict]:
    """Score each prompt's responses with ArmoRM.

    Args:
        prompt_texts: list[P] of biased prompts x'.
        response_groups: list[P], each a list of response strings.
        return_heads: if True, also return per-objective head vectors.

    Returns: list[P] of {"score": [r1, ..., rn], "heads": [[h1..h19], ...]?}
    """
    out: list[dict] = []
    for prompt, responses in zip(prompt_texts, response_groups):
        if not responses:
            out.append({"score": [], **({"heads": []} if return_heads else {})})
            continue
        r = _score(
            reward,
            prompt,
            responses,
            batch_size=batch_size,
            return_heads=return_heads,
        )
        out.append(r)
    return out
