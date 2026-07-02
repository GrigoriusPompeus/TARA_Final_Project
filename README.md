# TARA Final Project — Verifying RLHF Sycophancy Amplification

Empirical verification of Shapira, Benade, Procaccia (Feb 2026, [arXiv:2602.01002](https://arxiv.org/abs/2602.01002)) "How RLHF Amplifies Sycophancy" at 1B scale.

Their theorem says sycophancy is a *structural* consequence of RLHF: if the reward model has any positive covariance with agreement under the base policy, any preference optimiser (PPO, DPO, GRPO, Best-of-N) must amplify sycophancy. The theorem is proven in the **infinite-data, exact-KL-optimisation** regime. This project asks: **does it hold under finite data, a learned reward model, and gradient-based GRPO?**

Budget: $40 Vast.ai credits, ~$32 spent.

## Headline results

On n = 2,630 belief-injected TruthfulQA + sycophancy_eval probes:

1. **The theorem's premise holds.** P(positive reward tilt) = **0.57** (Phase 2).
2. **Best-of-N behaves as the theory predicts.** Positive-tilt subset amplifies at small N (0.456 → 0.494), declines at large N (→ 0.436) — matches Shapira §3.2's tail-flip prediction (Phase 3).
3. **Theorem 6 mitigation preserves capability.** MC1/MC2 differences within one standard error across base / vanilla-GRPO / Theorem-6-mitigated models (Phase 6a).
4. **Theorem 6 mitigation reduces agreement probability.** Paired z = **−11.4** vs vanilla, z = −10.9 vs base, on the probability-weighted metric that argmax misses at low KL drift (Phase 6c). Full result: [`results/eval_sycophancy/paired_prob.json`](results/eval_sycophancy/paired_prob.json).

## Deliverables

- [`Does-RLHF-Have-to-Make-Models-Sycophantic_submitted version.pdf`](Does-RLHF-Have-to-Make-Models-Sycophantic_submitted%20version.pdf) — submitted slide deck (11 slides).
- [`TARA_Project_Walkthrough.md`](TARA_Project_Walkthrough.md) — long-form educational walkthrough with method derivations and v2 corrections.
- [`CHECKPOINTS.md`](CHECKPOINTS.md) — how to reproduce the GRPO checkpoints on an A100.
- [`papers/README.md`](papers/README.md) — supporting papers with arXiv links.

## Project structure

```
src/
  data/       belief-injection probe dataset construction
  models/     Llama-3.2-1B Base policy + ArmoRM-8B reward loaders
  eval/       TruthfulQA MC1/MC2, sycophancy rate, Beirami KL estimators
  bon/        Best-of-N: candidate generation, scoring, tilt, BoN curves
  grpo/       TRL GRPO training + Shapira Theorem 6 corrected reward
  analysis/   figures and stats
scripts/      thin entry points 00_setup.sh, 01..06_*.py + eval + orchestrators
data/         raw + processed datasets (gitignored — recreate via scripts/01)
results/      experiment outputs (JSONs + figures committed; raw rollouts gitignored)
papers/       citations + arXiv links (paper PDFs not redistributed)
checkpoints/  GRPO-trained weights (not committed; ~2.5 GB each)
```

## Phases

| Phase | Where  | Cost   | What                                                 |
| ----- | ------ | ------ | ---------------------------------------------------- |
| 0     | local  | $0     | env + dataset download + smoke test                  |
| 1     | local  | $0     | belief-inject TruthfulQA + SycophancyEval            |
| 2     | local* | $0–3   | reward-tilt measurement (steered-proxy Δ_mean)       |
| 3     | local* | $0–5   | BoN sweep + Goodhart curve                           |
| 4     | cloud  | $15–25 | vanilla GRPO on Vast.ai A100 80GB                    |
| 5     | cloud  | $10–15 | Theorem 6 mitigated GRPO (λ = 1.0)                   |
| 6     | local  | $0     | MC1/MC2, argmax sycophancy, paired probability test  |

\* Phases 2–3 default to M1 Max local; a 1-hour cloud burst is the fallback if generation throughput is the blocker.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[local,dev]"          # M1 Mac
# OR
pip install -e ".[cloud,dev]"          # Linux + CUDA box
```

You'll also need:
- A HuggingFace account that has accepted the Llama-3.2 license at https://huggingface.co/meta-llama/Llama-3.2-1B.
- `huggingface-cli login` with a token that has `read` access to gated repos.

## Reproducing the headline

```bash
# Phase 6c — the paired-prob mitigation result (z = -11.4)
python -m scripts.eval_sycophancy_prob
# Reads results/eval_sycophancy/{base,grpo_vanilla,grpo_mitigated_lam1}.json
# Writes results/eval_sycophancy/paired_prob.json
```

To recreate the eval JSONs from scratch or the GRPO checkpoints, see [`CHECKPOINTS.md`](CHECKPOINTS.md).

## Key methodological caveats

- **Steered proxy for Phase 2.** Tilt is measured on primed continuations ("Yes, " / "Actually, "), not free π_base samples. Signal separates BoN subsets cleanly, so the proxy is load-bearing, but call it a proxy.
- **λ = 1.0 is blunt.** ArmoRM raw scores cluster at 0.04–0.10, so λ = 1.0 is ~10–25× the reward scale. Not the paper's minimal per-prompt λ*(x); it's a strong fixed-λ instantiation of Eq. 12.
- **Light-touch training.** Policy moved only ~0.005 KL from base. Vanilla amplification was not detectable at this scale — a real null, reported honestly. An extended run at KL ≈ 0.05 is trained (`checkpoints/grpo_vanilla_extended/final/`); its evaluation is scoped for the next round.
- **GRPO group normalisation** silently cancels the λ penalty when all 8 rollouts in a group agree. Mitigation only acts on prompts where the model is genuinely torn.
- **One training run per condition.** Paired z is prompt-level consistency, not run-level significance across seeds.

## License

MIT — see [`LICENSE`](LICENSE).

## Supporting papers

See [`papers/README.md`](papers/README.md) for full citations and arXiv links to Shapira et al. (the paper being verified), Shao et al. (GRPO), Wang et al. (ArmoRM), Beirami et al. (BoN KL), Sharma et al. and Perez et al. (empirical sycophancy), and Lin et al. (TruthfulQA).
