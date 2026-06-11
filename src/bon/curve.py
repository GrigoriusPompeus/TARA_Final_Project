"""Best-of-N optimization-pressure curve.

For each biased prompt x' we already have:
    - a pool of N_max candidate responses sampled from the (few-shot wrapped)
      base policy
    - per-response ArmoRM score r(x', y_i)
    - per-response agreement indicator A(x', y_i) ∈ {0, 1}

Best-of-N selects the highest-scoring of the first N samples. The sycophancy
rate of the BoN policy at level N on a set of prompts is:

    syc_rate(N) = E_{x' ~ D_false} [ A(x', y_BoN_N) ]

By Shapira Theorem 3, this rate increases in N if and only if Δ̂_mean(x') > 0
on those prompts. So we partition prompts by sign of Δ̂_mean (from Phase 2)
and plot syc_rate(N) for both subsets — expecting positive-tilt prompts to
RISE and negative-tilt prompts to FALL with N. This is the signature
verification of the covariance theorem.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass
class BoNPoint:
    n: int
    syc_rate: float
    mean_reward: float          # average max-N reward across prompts (proxy)
    correct_rate: float | None  # gold-truth metric if available
    n_prompts: int


def bon_select_indices(
    rewards: list[float],
    n: int,
    *,
    seed: int = 42,
    n_bootstrap: int = 1,
) -> list[int]:
    """Return n_bootstrap indices, each = argmax over a fresh random subset of n.

    For deterministic comparison across n values, we use a single shuffled
    permutation seeded per prompt: BoN(N) takes argmax over rewards[perm[:N]].
    """
    if not rewards:
        return []
    out: list[int] = []
    rng = random.Random(seed)
    for b in range(n_bootstrap):
        perm = list(range(len(rewards)))
        rng.shuffle(perm)
        subset = perm[: min(n, len(perm))]
        best_local = max(subset, key=lambda i: rewards[i])
        out.append(best_local)
    return out


def sweep(
    per_prompt_pool: list[dict],
    n_values: list[int],
    *,
    seed: int = 42,
    n_bootstrap: int = 8,
) -> list[BoNPoint]:
    """Sweep BoN over `n_values` against per-prompt pools.

    Each entry of per_prompt_pool:
        {
          "source_id": str,
          "rewards": [r_1, ..., r_M],
          "agree": [A_1, ..., A_M],
          "correct": [C_1, ..., C_M] | None,   # 1 if y_i contains gold answer
        }

    Returns a list aligned with n_values.
    """
    points: list[BoNPoint] = []
    for n in n_values:
        syc_total = 0.0
        rew_total = 0.0
        cor_total = 0.0
        cor_seen = 0
        n_used = 0
        for pi, entry in enumerate(per_prompt_pool):
            rewards = entry["rewards"]
            if not rewards or len(rewards) < 1:
                continue
            picks = bon_select_indices(
                rewards,
                n,
                seed=seed + 1000 * pi,
                n_bootstrap=n_bootstrap,
            )
            if not picks:
                continue
            syc_local = sum(entry["agree"][i] for i in picks) / len(picks)
            rew_local = sum(rewards[i] for i in picks) / len(picks)
            syc_total += syc_local
            rew_total += rew_local
            if entry.get("correct"):
                cor_local = sum(entry["correct"][i] for i in picks) / len(picks)
                cor_total += cor_local
                cor_seen += 1
            n_used += 1

        if n_used == 0:
            points.append(BoNPoint(n=n, syc_rate=float("nan"), mean_reward=float("nan"),
                                   correct_rate=None, n_prompts=0))
            continue
        points.append(
            BoNPoint(
                n=n,
                syc_rate=syc_total / n_used,
                mean_reward=rew_total / n_used,
                correct_rate=(cor_total / cor_seen) if cor_seen else None,
                n_prompts=n_used,
            )
        )
    return points


def partition_by_tilt(
    pool: list[dict],
    tilts: dict[str, float],
) -> dict[str, list[dict]]:
    """Split per_prompt_pool into positive/negative tilt subsets using Phase-2 tilts."""
    pos: list[dict] = []
    neg: list[dict] = []
    other: list[dict] = []
    for e in pool:
        sid = e["source_id"]
        t = tilts.get(sid)
        if t is None or t != t:  # NaN
            other.append(e)
        elif t > 0:
            pos.append(e)
        else:
            neg.append(e)
    return {"positive": pos, "negative": neg, "untagged": other}
