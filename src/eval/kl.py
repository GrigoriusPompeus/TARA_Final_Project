"""KL-divergence estimators for the BoN policy.

Beirami et al. (ICML 2025, arXiv:2401.01879) prove that the commonly-cited
formula

    KL(π^(n) ‖ π_ref)  ≈  log(N) − (N−1)/N         (Eq. 4)

is in fact only an UPPER BOUND on the true KL (Theorem 3.1). They propose a
tighter estimator (Eq. 25) that closely matches the exact KL across many
reward distributions:

    D̂_KL(ε_n) := d_n(ε_n)
    d_n(ε) = (1 − ε)^n [log n + (n−1) log(1 − ε) − (n−1)/n]
           + (1 − (1 − ε)^n) log( (1 − (1 − ε)^n) / ε )

where ε_n = π_ref(y_max | x) under the best-of-n selected y_max.

We expose both formulas plus a helper that takes a list of `eps_n` values
(typically estimated by Monte Carlo across prompts) and returns the proposed
KL estimator's expectation.
"""

from __future__ import annotations

import math


def kl_upper_bound(n: int) -> float:
    """Eq. 4: KL_n_tilde = log(n) − (n-1)/n. ONLY an upper bound. Use sparingly."""
    if n < 1:
        raise ValueError("n must be >= 1")
    if n == 1:
        return 0.0
    return math.log(n) - (n - 1) / n


def proposed_kl_estimator(n: int, eps: float) -> float:
    """Eq. 25 of Beirami et al.

    eps is the probability mass on the best-of-n selected outcome under π_ref.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if n == 1:
        return 0.0
    if eps <= 0.0:
        # numerical floor; treat as the upper bound
        return kl_upper_bound(n)
    if eps >= 1.0:
        return 0.0
    one_minus_eps_n = (1.0 - eps) ** n
    survives = 1.0 - one_minus_eps_n
    term1 = one_minus_eps_n * (math.log(n) + (n - 1) * math.log(1.0 - eps) - (n - 1) / n)
    if survives <= 0.0:
        term2 = 0.0
    else:
        term2 = survives * math.log(survives / eps)
    return term1 + term2


def expected_proposed_kl(n: int, eps_samples: list[float]) -> float:
    """Monte-Carlo average of the proposed estimator over sample eps_n values.

    `eps_samples[i]` = π_ref(y_max_i | x_i) from a separate base-policy
    sampling pass. The mean over many prompts approximates the cross-prompt
    KL divergence used by alignment evaluations.
    """
    if not eps_samples:
        return 0.0
    vals = [proposed_kl_estimator(n, e) for e in eps_samples]
    return sum(vals) / len(vals)


def win_rate_upper_bound(n: int) -> float:
    """Theorem 5.3 of Beirami: W_r(π^(n) || π_ref)  <=  n / (n+1)."""
    if n < 1:
        raise ValueError("n must be >= 1")
    return n / (n + 1)
