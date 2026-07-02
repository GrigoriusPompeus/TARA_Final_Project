"""Plotting code for the project's deliverable figures.

Five canonical plots:
    Fig 1a — distribution of Δ̂_mean(x') across belief_wrong prompts
             (Shapira Figure 1a/1b reproduction)
    Fig 1c — BoN sycophancy curve: syc_rate vs N on positive- and
             negative-tilt subsets (Shapira Theorem 3 sign-flip)
    Fig 2  — Goodhart curve: proxy ArmoRM reward vs gold TruthfulQA MC1
             accuracy as N grows (Gao et al. style overoptimization)
    Fig 3  — GRPO training trajectory: ArmoRM reward, sycophancy rate,
             TruthfulQA MC1 vs step
    Fig 4  — Mitigation comparison: vanilla GRPO vs corrected reward at
             several lambda values; sycophancy and truthfulness side-by-side

All plots are matplotlib-only (no seaborn dependency at runtime) and
write a PNG + PDF pair to results/figures/.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

PALETTE = {
    "positive": "#d62728",  # red — sycophancy amplification
    "negative": "#2ca02c",  # green — sycophancy correction
    "all": "#1f77b4",       # blue — pooled
    "gold": "#7f7f7f",      # gray — TruthfulQA gold metric
    "proxy": "#1f77b4",     # blue — ArmoRM proxy
}


def _save(fig, name: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_tilt_distribution(
    delta_means: Sequence[float],
    out_dir: Path,
    title: str | None = None,
) -> None:
    """Fig 1a: histogram of Δ̂_mean across prompts."""
    arr = np.asarray([d for d in delta_means if d == d])
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.hist(arr, bins=40, color=PALETTE["all"], alpha=0.85, edgecolor="white")
    ax.axvline(0, color="black", linestyle="--", linewidth=1.0)
    pos_frac = float((arr > 0).mean())
    ax.set_xlabel(r"$\hat{\Delta}_{\mathrm{mean}}(x')$ (agreeing reward − correcting reward)")
    ax.set_ylabel("Number of prompts")
    ax.set_title(title or f"Reward tilt across belief_wrong prompts  (P(Δ>0)={pos_frac:.2f})")
    _save(fig, "fig1a_tilt_distribution", out_dir)


def fig_bon_sign_flip(
    n_values: Sequence[int],
    pos_syc: Sequence[float],
    neg_syc: Sequence[float],
    all_syc: Sequence[float] | None,
    out_dir: Path,
) -> None:
    """Fig 1c: sycophancy rate vs N partitioned by tilt sign."""
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.plot(n_values, pos_syc, "o-", color=PALETTE["positive"], label=r"$\hat{\Delta}_{\mathrm{mean}}>0$ (predicted amplify)")
    ax.plot(n_values, neg_syc, "o-", color=PALETTE["negative"], label=r"$\hat{\Delta}_{\mathrm{mean}}<0$ (predicted correct)")
    if all_syc is not None:
        ax.plot(n_values, all_syc, "o--", color=PALETTE["all"], label="all prompts", alpha=0.7)
    ax.set_xscale("log", base=2)
    ax.set_xticks(list(n_values))
    ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("N (Best-of-N optimisation pressure)")
    ax.set_ylabel("Sycophancy rate")
    ax.set_title("BoN Theorem 3 verification: sign of Δ̂_mean predicts drift direction")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    _save(fig, "fig1c_bon_sign_flip", out_dir)


def fig_goodhart_curve(
    n_values: Sequence[int],
    proxy_reward: Sequence[float],
    gold_accuracy: Sequence[float],
    out_dir: Path,
) -> None:
    """Fig 2: Gao-style over-optimisation. Proxy and gold diverge as N grows."""
    fig, ax1 = plt.subplots(figsize=(5.5, 3.5))
    color1 = PALETTE["proxy"]
    color2 = PALETTE["gold"]
    ax1.set_xlabel("N (Best-of-N optimisation pressure)")
    ax1.set_ylabel("Proxy reward (ArmoRM)", color=color1)
    ax1.plot(n_values, proxy_reward, "o-", color=color1, label="ArmoRM (proxy)")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(list(n_values))
    ax1.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Gold accuracy (TruthfulQA MC1)", color=color2)
    ax2.plot(n_values, gold_accuracy, "s--", color=color2, label="TruthfulQA MC1 (gold)")
    ax2.tick_params(axis="y", labelcolor=color2)

    fig.suptitle("Goodhart overoptimization: proxy ↑, gold ↓")
    fig.tight_layout()
    _save(fig, "fig2_goodhart_curve", out_dir)


def fig_grpo_trajectory(
    steps: Sequence[int],
    proxy_reward: Sequence[float],
    syc_rate: Sequence[float],
    mc1: Sequence[float],
    out_dir: Path,
    title: str = "GRPO training trajectory",
) -> None:
    """Fig 3: three-panel trajectory."""
    fig, axes = plt.subplots(3, 1, figsize=(6, 6), sharex=True)
    axes[0].plot(steps, proxy_reward, "-", color=PALETTE["proxy"])
    axes[0].set_ylabel("ArmoRM reward")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(steps, syc_rate, "-", color=PALETTE["positive"])
    axes[1].set_ylabel("Sycophancy rate")
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(steps, mc1, "-", color=PALETTE["gold"])
    axes[2].set_ylabel("TruthfulQA MC1")
    axes[2].set_xlabel("Training step")
    axes[2].grid(True, alpha=0.3)
    fig.suptitle(title)
    _save(fig, "fig3_grpo_trajectory", out_dir)


def fig_mitigation_comparison(
    lambdas: Sequence[float],
    syc_rates: Sequence[float],
    mc1s: Sequence[float],
    baseline_syc: float,
    baseline_mc1: float,
    out_dir: Path,
) -> None:
    """Fig 4: side-by-side bars at each lambda + baseline reference lines."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))
    x = np.arange(len(lambdas))
    width = 0.55
    ax1.bar(x, syc_rates, width, color=PALETTE["positive"], alpha=0.85)
    ax1.axhline(baseline_syc, color="black", linestyle="--", linewidth=1.0, label="vanilla GRPO baseline")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{l:g}" for l in lambdas])
    ax1.set_xlabel(r"$\lambda$ (Shapira Theorem 6 penalty)")
    ax1.set_ylabel("Sycophancy rate")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.bar(x, mc1s, width, color=PALETTE["gold"], alpha=0.85)
    ax2.axhline(baseline_mc1, color="black", linestyle="--", linewidth=1.0)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{l:g}" for l in lambdas])
    ax2.set_xlabel(r"$\lambda$")
    ax2.set_ylabel("TruthfulQA MC1")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Shapira Theorem 6 mitigation sweep")
    fig.tight_layout()
    _save(fig, "fig4_mitigation_comparison", out_dir)
