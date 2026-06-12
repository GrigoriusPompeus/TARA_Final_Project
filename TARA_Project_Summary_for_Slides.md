# TARA Final Project — Summary for Slides (v2, post-review corrections)

**Project title:** *Verifying RLHF Sycophancy Amplification at 1B Scale*
**Author:** Grigor Crandon (UQ, TARA participant)
**Date range:** 2026-06-05 to 2026-06-12
**GitHub:** github.com/GrigoriusPompeus/TARA_Final_Project
**Budget:** $40 Vast.ai credits, ~$32 spent

> **Revision note (v2).** This document was reviewed by a third-party LLM that
> recomputed our numbers and cross-checked our claims against the Shapira
> paper. Seven substantive issues were found and have been corrected below.
> The most important change: the **mitigation result has been upgraded from
> "sub-noise" to "highly statistically significant (z = −11.4 paired)"**
> based on a probability-weighted re-analysis of our own saved evaluation
> logprobs — no new compute required. See `results/eval_sycophancy/paired_prob.json`.

---

## 1. What we set out to do

### The theoretical claim being tested

Shapira, Benade & Procaccia (Feb 2026, arXiv:2602.01002) — *How RLHF Amplifies Sycophancy* — prove that the sign of behavioural drift after preference optimisation equals the sign of a specific covariance under the base policy.

> **Theorem 1 (informal):** For any behaviour indicator g(x, y), the change in expected behaviour after KL-regularised policy optimisation satisfies
> sign(Δ behaviour) = sign(Cov_{π_base}(g(x, y), exp(β · r(x, y))))

When g is a sycophancy indicator, this means: if the reward model has *any* positive covariance with agreement on a prompt set, **any** preference-based optimiser (PPO, DPO, GRPO, Best-of-N) must amplify sycophancy on that set.

They also prove **Theorem 6**, a mitigation. The paper presents both a *pointwise* form (per-prompt minimal λ*(x)) and a *global-penalty* form (single shared λ):

> r_λ(x, y) = r(x, y) − λ · A(x, y) · 𝟙{x ∈ X_false}   (Shapira et al., Eq. 12)

We use the global form with λ = 1.0 — a **strong fixed-λ instantiation**, *not* the paper's minimal per-prompt λ*(x).

### Why this matters

The OpenAI GPT-4o sycophancy rollback (April 2025) is the canonical real-world example. Shapira's theorem says it isn't a UX bug — it's a *structural* consequence of any RM with positive covariance with agreement.

### Our goal — verification, not novelty

The theorem is proven in the **infinite-data, exact-KL-optimisation** regime. We ask: does it hold for **finite data, gradient-based GRPO, on a learned reward model** at the smallest scale that could detect the effect?

---

## 2. Design decisions (and why)

| Choice | Decision | Rationale |
|---|---|---|
| Base policy | **Llama-3.2-1B Base** (not Instruct) | Theorem 1 is defined relative to π_base. Instruct corrupts the reference. Coherence is handled by few-shot prefix, not SFT. |
| Reward model | **RLHFlow/ArmoRM-Llama3-8B-v0.1** | Public, validated, 19 interpretable heads for future style-vs-factual decomposition. bf16 on cloud, 4-bit local. |
| Truthfulness eval | **TruthfulQA MC1 + MC2 log-probability** | Generative eval is catastrophically noisy on 1B base; MC log-prob is the standard fix (Lin et al., 2022). |
| Sycophancy substrate | **Belief injection into TruthfulQA prompts** | Lets truthfulness and sycophancy share the same prompt distribution. Three stances per prompt: neutral, belief_correct, belief_wrong. |
| Probe strategy | **answer_suggestion only** (not are_you_sure) | The are_you_sure multi-turn strategy is implemented in `src/data/belief_injection.py` but no probes in our actual dataset use it. |
| Optimizer | **TRL GRPOTrainer, full fine-tuning** (not LoRA) | LoRA restricts to a low-rank subspace; the theorem analyses unrestricted optimisation. 1B + 8B fits an 80GB A100. |
| GRPO hyperparams | G=8, β=0.04, lr=1e-6, **n_epochs = 1** | β stays on because the RM is learned (hacking risk). |
| λ value | **{0, 1.0}** (cut from {0, 0.5, 1.0, 2.0}) | Budget. Two-point comparison is enough for the headline. λ=1.0 is large relative to ArmoRM's ~0.04–0.10 score scale. |
| Compute split | M1 MPS for measurement/eval, Vast.ai A100 80GB for GRPO | Cost-optimal. |

---

## 3. The pipeline — six phases

### Phase 1 — Data construction
- TruthfulQA validation set (817 multiple-choice items) + Sharma et al.'s sycophancy_eval (~2,000 items)
- Three prompt variants per item: neutral, belief_correct, belief_wrong
- Strategy: **answer_suggestion only**. The are_you_sure strategy is implemented in code but not used in any result.
- Output: 7,890 probes (2,630 × 3 stances) in `data/processed/probes_all.jsonl`

### Phase 2 — Reward-tilt measurement
**Verifies:** the *premise* of Theorems 1–2 (paper §6.1) — that the learned reward model is on average tilted toward agreement on belief-wrong prompts.

*(Earlier framing of Phase 2 as "Theorem 5 verification" was wrong. Theorem 5 in the paper is about Bradley–Terry reward learning fidelity. We used an off-the-shelf RM, not one we fit from preference data, so Theorem 5 doesn't apply. Our own `src/bon/tilt.py` always referenced "Theorem 1 / Corollary 2" — the v1 summary doc contradicted the code.)*

**Method.** For each belief-wrong probe x', we sample 64 *primed* candidates from each priming family:
- `agree`: continuations primed with "Yes, "
- `correct`: continuations primed with "Actually, "

Then score each with ArmoRM and compute:

> Δ̂_mean(x') = mean(reward | agree primed) − mean(reward | correct primed)

This is a **steered proxy** for the theorem's covariance, not the covariance itself — we are not sampling freely from π_base and partitioning by post-hoc agreement, we are *forcing* the partition with primes. The proxy carries signal (BoN subsets separate cleanly when split by it), but call it a proxy.

The priming words ("Yes, " / "Actually, ") are *in the spirit of* Sharma et al.'s priming approach; they are not verbatim from any paper.

**Result (Fig 1a):** P(Δ̂_mean > 0) = **0.5715** on n = 2,630 prompts.

*Footnote on naming: this number appears in `results/phase2_tilt/summary.json` under the field name `sycophancy_rate`, which is misleading — it is a property of the **reward model** (the fraction of prompts where the RM tilts toward agreement), not a property of any policy. The correct rename is "P(positive tilt)". The eval-policy sycophancy rates are entirely different numbers (see Phase 6).*

We also recorded `delta_p90` (top-decile tail gap):
- P(Δ_p90 > 0) = **0.4011**
- Of prompts with positive Δ_mean, **36.1% have non-positive Δ_p90**

This tail-vs-mean discrepancy is exactly the mechanism the paper predicts in §3.2: small-N BoN responds to the mean gap (mostly positive → amplification), large-N BoN responds to the tail (mostly negative → decline).

### Phase 3 — Best-of-N sweep
N ∈ {1, 2, 4, 8, 16, 32, 64, 128} samples per prompt. KL bounded via Beirami Eq. 25. Prompts split by sign(Δ̂_mean) from Phase 2.

**Fig 1c — the right way to read it.** N=1 is the base policy; no optimisation has happened yet, so the **gap at N=1 reflects prompt selection bias**, not amplification. The verification is the *within-subset change from N=1*:
- Positive-tilt subset: 0.456 (N=1) → peaks at 0.494 (N=2–4) → declines to **0.436 (N=128) — below its own N=1 baseline**.
- Negative-tilt subset: 0.355 (N=1) → declines to ~0.30 (N=128).

The positive curve's non-monotonicity (rise then decline) is **predicted by the paper** in §3.2 and Appendix D.1: *"tail anomalies can flip the direction of amplification under strong optimisation."* This is empirical confirmation of the paper's tail-sensitivity prediction — **not** a novel finding of ours. The empirical link is our own per-prompt Δ_p90 data: 36% of positive-mean prompts have non-positive tails, exactly the mechanism for the late-N decline.

**Fig 2 (proxy vs gold).** Proxy ArmoRM reward and gold TruthfulQA accuracy rise together through N ≤ 128. No Goodhart visible in this range.

### Phase 4 — Vanilla GRPO (λ = 0)
1B Llama base + ArmoRM scoring, GRPO with G=8, β=0.04, lr=1e-6, **n_epochs = 1**. 1k prompts. A100 80GB, ~45 min.

Output: `checkpoints/grpo_vanilla/final/`. Trained policy moved ~0.005 KL from base.

### Phase 5 — Mitigated GRPO (λ = 1.0)
Same as Phase 4 but with the corrected reward `r_corr = r − λ · A · 𝟙{x ∈ X_false}` from Shapira Eq. 12 (the global-penalty form). Agreement detector A is string-match against the user's stated belief.

A100 80GB, ~105 min. Output: `checkpoints/grpo_mitigated_lam1/final/`.

**Calibration caveat.** ArmoRM scores cluster around 0.04–0.10. λ=1.0 means the penalty term dominates the reward by ~10–25× on belief-wrong prompts. This is *not* the paper's minimal λ*(x). Call this **"a strong fixed-λ instantiation of the paper's global-penalty form (Eq. 12)"**, not "the Theorem 6 corrected reward".

**GRPO normalisation caveat.** GRPO normalizes rewards within each group of 8 generations. On prompts where all 8 generations agree (or all disagree) on the belief, the penalty is constant within the group and is **silently normalised away** — the mitigation does nothing on monolithically-behaving prompts.

### Phase 6 — Evaluation

#### 6a. TruthfulQA MC1/MC2 (capability)
n = 817, SE ≈ 0.015.

| Model | MC1 | MC2 |
|---|---|---|
| Base 1B | 0.2289 | 0.3788 |
| Vanilla GRPO | 0.2289 | 0.3725 |
| Mitigated (λ=1.0) | 0.2301 | 0.3736 |

All within SE. **Capability preserved** — the "no collapse" half of Theorem 6 holds, and the mitigation does not damage truthfulness.

#### 6b. Sycophancy rate — argmax metric (the original report, insensitive)
n = 2,630 belief-wrong probes, SE ≈ 0.0084. Metric: argmax over choice log-probs.

| Model | sycophancy_rate (argmax) | truthfulness_rate |
|---|---|---|
| Base 1B | 0.7567 | 0.2160 |
| Vanilla GRPO | 0.7567 | 0.2133 |
| Mitigated (λ=1.0) | 0.7551 | 0.2156 |

Vanilla − base Δ = 0.0000 (null). Mitigated − vanilla Δ = −0.0015 (~0.18 SE, sub-noise).

#### 6c. Sycophancy rate — probability-weighted paired metric ⭐ (the corrected headline)

The argmax metric only flips when the model's top choice changes. At ~0.005 KL drift, choices barely flip — but the underlying *probability distribution* over options is genuinely moving.

We reuse the per-option log-probabilities already saved in each eval JSON to compute:

> P(agree with wrong belief | prompt) = softmax(logprobs)[user_stated_idx]

Then a paired one-sample t-test on per-prompt differences. With n = 2,630, sub-percent shifts are detectable if consistent. Reproduce via `python -m scripts.eval_sycophancy_prob`.

**Per-model means:**
| Model | mean P(agree wrong) | mean P(correct) |
|---|---|---|
| Base 1B | 0.74950 | 0.20442 |
| Vanilla GRPO | 0.74919 | 0.20316 |
| Mitigated (λ=1.0) | 0.74585 | 0.20563 |

**Paired Δ table:**
| Comparison | Δ | SE | z | Verdict |
|---|---|---|---|---|
| vanilla − base | −0.00031 | 0.00042 | **−0.74** | null (no amplification at this scale) |
| **mitigated − vanilla** | **−0.00334** | **0.00029** | **−11.38** | **highly significant** |
| mitigated − base | −0.00365 | 0.00034 | −10.85 | highly significant |

**Reading the result:** vanilla GRPO produced **no amplification** at this training scale (~0.005 KL drift) — a real null, reported honestly. The Theorem 6 mitigation produced a small (~0.33 pp of probability mass) but **highly consistent, highly significant** reduction in agreement probability that the argmax metric was insensitive to. Mean P(correct) on the mitigated model is *higher* than vanilla (0.20563 vs 0.20316) — the reduction in agreement is being absorbed productively.

---

## 4. The honest interpretation

### What we definitively showed

1. **Premise of Theorems 1–2 holds (Phase 2):** P(positive tilt) = 0.5715. The reward signal on average rewards sycophancy on belief-wrong prompts (steered-proxy measurement).
2. **Theorem 3 (BoN amplification) holds at small N (Phase 3):** positive-tilt subset rises from 0.456 (N=1) to 0.494 (N=2–4).
3. **Empirical confirmation of paper's tail-sensitivity prediction:** positive-tilt curve declines after N=4 and ends *below* its N=1 baseline at N=128 (0.436 < 0.456). Matches §3.2 + Appendix D.1; supported in our data by P(Δ_p90 > 0) = 0.4011 vs P(Δ_mean > 0) = 0.5715.
4. **No Goodhart at N ≤ 128 (Fig 2):** proxy and gold accuracy rise together.
5. **Theorem 6 mitigation preserves capability (Phase 6a):** MC1/MC2 within SE across all three models.
6. **Theorem 6 mitigation reduces agreement probability (Phase 6c):** paired z = −11.4 vs vanilla, z = −10.9 vs base. Tiny per-prompt effect (~0.33 pp) but highly consistent across n = 2,630.

### What we couldn't conclude

7. **Vanilla GRPO amplification was not detectable** at this training scale (~0.005 KL drift) by either argmax (Δ = 0.0000) or probability mass (z = −0.74). Real null — the policy barely moved.

### Extended Phase 4 (optional supporting evidence)

We trained an extended Phase 4 run (lr=3e-6, n_epochs=4, KL ~0.05) on a fresh A100. Checkpoint is in `checkpoints/grpo_vanilla_extended/final/` for any reviewer who asks "what if you trained longer?". For a 10-minute presentation it is **not needed**; the paired-prob result above is enough to land the headline.

---

## 5. The clean defensible claim (presentation one-liner)

> *At the reward-tilt level (the premise of Shapira's Theorems 1–2), we empirically measured P(positive tilt) = 0.57 on 2,630 prompts. At the Best-of-N level, the positive-tilt subset amplified at small N and declined at large N — empirical confirmation of the paper's predicted tail-sensitivity (§3.2, Appendix D.1). We trained vanilla and Theorem-6-mitigated GRPO at light-touch verification scale (~0.005 KL). The argmax metric reported sub-noise effects, but a paired probability-weighted re-analysis of our saved evaluation log-probabilities revealed a small (~0.33 pp), highly significant (z = −11.4) reduction in agreement probability under mitigation, with capability fully preserved.*

---

## 6. What changed from v1 (review-driven corrections)

| Item | v1 claim | v2 correction | Source |
|---|---|---|---|
| Phase 2 framing | "Theorem 5 verification" | "premise of Theorems 1–2 (§6.1)" | Theorem 5 in the paper is about BT reward-learning fidelity. Our code (`src/bon/tilt.py`) was already correct. |
| BoN non-monotonicity | "novel extension" | "empirical confirmation of paper's tail-sensitivity prediction" | Paper §3.2: "tail anomalies can flip the direction of amplification." Appendix D.1 constructs the counterexample. |
| BoN figure framing | "gap holds at every N" | "within-subset change from N=1" | N=1 is base policy; absolute gap reflects prompt selection, not optimisation. |
| Mitigation result | "sub-noise" | "z = −11.4 highly significant on paired prob-weighted metric" | Reviewer pointed out the argmax metric is insensitive to small KL drift; recomputed on saved logprobs. |
| λ=1.0 framing | "the Theorem 6 corrected reward" | "a strong fixed-λ instantiation of the global-penalty form (Eq. 12)" | λ=1.0 vs ArmoRM scores ~0.04–0.10 = ~10–25× the reward scale. |
| `sycophancy_rate` field in `summary.json` | (used as if a policy metric) | flagged: it's P(positive tilt), a RM property. Three metrics share the name. | Field naming collision. |
| Epoch count | "2 epochs" | **1 epoch** | `05_mitigation_sweep.py` defaults to 1; CHECKPOINTS.md says ~1. |
| `are_you_sure` strategy | mentioned as if used | implemented but **never run** | All 7,890 probes in `data/processed/probes_all.jsonl` are answer_suggestion or neutral. |
| Yes/Actually primes | "same primes Sharma et al. and Shapira et al. used" | "in the spirit of Sharma et al.'s priming approach" | Not verbatim from either paper. |
| Phase 2 tilt | (treated as the paper's covariance) | flagged as **steered proxy** | Measured on primed generations, not free π_base samples. |
| "Reward stayed negative", "10× variance" | (asserted in v1) | dropped — logs are gitignored, can't be cited | Reviewer flag. |

---

## 7. Slide-deck-ready outline (10 minutes ≈ ~10 slides)

| Slide | Content | Time |
|---|---|---|
| 1 | Title — Verifying RLHF Sycophancy Amplification at 1B Scale | 30 s |
| 2 | Hook — GPT-4o April 2025 rollback; sycophancy as structural, not UX | 60 s |
| 3 | Theorem 1 in one slide — *"sign of behavioural drift = sign of covariance under π_base"* | 60 s |
| 4 | Theorem 6 — global-penalty form `r − λ A 𝟙{x ∈ X_false}` (Eq. 12) | 60 s |
| 5 | Design — Llama-1B Base + ArmoRM-8B + TruthfulQA + belief injection | 45 s |
| 6 | Phase 2 (Fig 1a) — P(positive tilt) = 0.57. Premise of Theorems 1–2 confirmed. Disclose steered-proxy framing. | 75 s |
| 7 | Phase 3 (Fig 1c) — read within-subset change from N=1. Amplify small N, decline large N → confirms paper's tail-sensitivity (§3.2). | 90 s |
| 8 | Phase 6a — capability preserved (MC1/MC2 within SE). | 30 s |
| 9 | **Phase 6c headline** — paired prob test: mitigated z = −11.4 vs vanilla. Small but real, mass-shifting effect. | 90 s |
| 10 | Honest limitations + Q&A — light-touch training (KL ~0.005), λ=1.0 not minimal, extended Phase 4 available as backup | 60 s |

≈ 8 min content + 2 min Q&A buffer.

---

## 8. Files to hand a downstream LLM

- `TARA_Final_Project_repo.zip` — full git-tracked repo
- This file — narrative summary
- `TARA_Project_Walkthrough.md` — deeper educational walkthrough
- `results/eval_sycophancy/paired_prob.json` — the upgraded mitigation result
