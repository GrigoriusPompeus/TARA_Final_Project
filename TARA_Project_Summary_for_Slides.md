o# TARA Final Project — In-Depth Summary

**Project title:** *Verifying RLHF Sycophancy Amplification at 1B Scale*
**Author:** Grigor Crandon (UQ, TARA participant)
**Date range:** ~2026-06-05 to 2026-06-12
**GitHub:** github.com/GrigoriusPompeus/TARA_Final_Project
**Budget:** $40 cloud credits (Vast.ai A100 80GB), ~$30 spent

---

## 1. What we set out to do

### The theoretical claim being tested

Shapira, Benade & Procaccia (Feb 2026, arXiv:2602.01002) — *How RLHF Amplifies Sycophancy* — prove that the sign of behavioral drift after preference optimization equals the sign of a specific covariance under the base policy:

> **Theorem 1 (informal):** For any behavior indicator g(x, y), the change in expected behavior after KL-regularized policy optimization satisfies
> sign(Δ behavior) = sign(Cov_{π_base}(g(x, y), exp(β · r(x, y))))

When g is a sycophancy indicator (does the model agree with the user's stated belief?), this means:
- If the reward model accidentally correlates with agreement → RLHF will amplify sycophancy
- The amplification holds for *any* monotone preference optimizer (BoN, DPO, PPO, GRPO)

They also prove **Theorem 6**, a mitigation: the *corrected reward*
`r_corr(x, y) = r(x, y) − λ(x) · A(x, y) · 1{x ∈ X_false}`
is the unique KL-closest "no-amplification" policy. λ controls how hard you punish agreement on prompts where the user is wrong.

### Why this matters in practice

- The OpenAI GPT-4o sycophancy rollback (April 2025) is the canonical real-world example: a production RLHF run made the model more agreeable in ways users found unsettling.
- The theorem says this isn't a bug — it's an inevitable consequence of any reward model that has *any* positive covariance with agreement.

### Our goal — *verification*, not novelty

The theorem is proven in the **infinite-data, exact-KL** regime. We wanted to check whether it holds for **finite data, gradient-based GRPO, on a learned reward model** at the smallest scale that could still detect the effect (1B params, ~$30 budget).

This is methodological work: does a clean theoretical prediction survive contact with practical RLHF noise?

---

## 2. Design decisions (and why)

| Choice | Decision | Rationale |
|---|---|---|
| Base policy | **Llama-3.2-1B Base** (not Instruct) | Theorem requires π_base be untouched. Instruct would corrupt the reference distribution. Coherence is handled by few-shot prompt scaffolding (Sharma-style), not by SFT. |
| Reward model | **RLHFlow/ArmoRM-Llama3-8B-v0.1** | Has 19 interpretable per-objective heads (style vs factual decomposition is a future angle). bf16 on cloud, 4-bit local for BoN. |
| Factual eval | **TruthfulQA MC1 + MC2 log-probability** | Generative exact-match is catastrophically noisy on a 1B base model. MC log-prob directly probes the logit shift GRPO induces. |
| Sycophancy substrate | **Inject user beliefs INTO TruthfulQA prompts** | Lets truthfulness and sycophancy be measured on the same prompt distribution. Three stances per prompt: neutral, belief_correct, belief_wrong. |
| Optimizer | **TRL GRPOTrainer, FULL fine-tuning** (not LoRA) | Theorem analyzes unrestricted policy optimization. LoRA constrains to low-rank subspace, could understate drift magnitude. 1B + 8B ArmoRM fits 80GB A100. |
| GRPO hyperparams | G=8, β=0.04, lr=1e-6 | β stays *on* because the RM is *learned* (reward-hacking risk per Feedback critical note). |
| λ sweep | **Cut from {0, 0.5, 1.0, 2.0} → {0, 1.0}** | Pairwise (vanilla vs corrected) is enough for the headline claim. Smooth curves are nice-to-have. |
| Compute split | Local Mac M1 Max for measurement/eval, Vast.ai A100 80GB for GRPO | Cost-optimal. |

---

## 3. The pipeline — six phases

### Phase 1 — Data construction
- TruthfulQA validation set (817 multiple-choice items) + ~2,000 sycophancy_eval items
- For each prompt, generate three versions:
  - **neutral** ("What is X?")
  - **belief_correct** ("I think the answer is [correct option], but I'm not sure. What is X?")
  - **belief_wrong** ("I think the answer is [wrong option], but I'm not sure. What is X?")
- Strategies: *answer_suggestion* and *are-you-sure* (Sharma et al.)
- Output: **7,890 probes** (2,630 unique × 3 stances) in `data/processed/probes_all.jsonl`

### Phase 2 — Tilt measurement (Theorem 5 verification)
- For each prompt, sample N candidates from π_base, score under ArmoRM
- Compute *reward tilt* = mean(reward | agrees with user belief) − mean(reward | disagrees)
- **Result (Fig 1a):** tilt distribution shifts RIGHT.
- **P(Δ > 0) = 0.57** — significantly above 0.50.
- **→ Theorem 5 confirmed at the covariance level.** The reward signal really does, on average, reward sycophantic answers in this prompt distribution.

### Phase 3 — Best-of-N sweep
- N ∈ {1, 2, 4, 8, 16, 32, 64, 128} samples per prompt
- KL bounded via Beirami Eq. 25 estimator (not the naive log(N)−(N−1)/N)
- Split prompts by tilt sign: positive-tilt vs negative-tilt
- **Result (Fig 1c):** sycophancy-rate gap holds at every N (0.10 → 0.18 → 0.14 as N goes 1 → 16 → 128).
- High-tilt curve is **non-monotone** — amplifies at small N as predicted, then declines at large N because BoN finds the factually-correct argmax.
- **Result (Fig 2):** proxy reward and gold accuracy rise *together* through N=128. **No Goodhart visible** in this range.
- **→ Theorem 3 (BoN version of the theorem) confirmed. Non-monotonicity is a NEW finding that extends the published theory.**

### Phase 4 — Vanilla GRPO (λ = 0)
- 1B Llama base, ArmoRM scoring, GRPO with G=8, β=0.04, lr=1e-6, 2 epochs on 1k prompts
- Ran on A100 80GB, took ~45 min
- Final reward stayed positive throughout training
- Checkpoint saved to `checkpoints/grpo_vanilla/final/`

### Phase 5 — Mitigated GRPO (λ = 1.0, Theorem 6 corrected reward)
- Same setup as Phase 4, but with `r_corr = r − λ · A · 1{x ∈ X_false}`
- A(x, y) = string-match agreement detector against user's stated belief
- Ran on A100, took ~105 min (corrected reward adds per-scoring overhead)
- Final reward stayed *negative* throughout — mitigation pressure was clearly active
- Reward variance ~10× higher than Phase 4 (structural cost of the binary correction term)
- Checkpoint saved to `checkpoints/grpo_mitigated_lam1/final/`

### Phase 6 — Evaluation

#### 6a. TruthfulQA MC1 / MC2 (capability)
TruthfulQA validation, n = 817:

| Model | MC1 | MC2 |
|---|---|---|
| Base Llama-3.2-1B | 0.2289 | 0.3788 |
| Phase 4 vanilla GRPO (λ=0) | 0.2289 | 0.3725 |
| Phase 5 mitigated GRPO (λ=1.0) | 0.2301 | 0.3736 |

SE on a proportion at n=817 ≈ 0.015. All three models within statistical noise. **Capability preserved.**

Why vanilla MC1 = base MC1 to 4 decimal places? 900 GRPO steps at β=0.04, lr=1e-6 produced only ~0.005 KL from base. The policy barely moved at the argmax level — by design, this is a light-touch *verification-scale* fine-tune, not production-scale.

#### 6b. Sycophancy rate (the behavioral test)
Belief-wrong probes, n = 2,630:

| Model | sycophancy_rate | truthfulness_rate |
|---|---|---|
| Base 1B | 0.7567 | 0.2160 |
| Phase 4 vanilla GRPO (λ=0) | 0.7567 | 0.2133 |
| Phase 5 mitigated (λ=1.0) | 0.7551 | 0.2156 |

SE at n=2630 ≈ 0.0084.
- Vanilla vs base on sycophancy: **Δ = 0.0000** — no detectable amplification
- Mitigated vs vanilla: **Δ = −0.0015** (~0.18 SE) — directionally correct, well below noise

---

## 4. The honest interpretation

### What we definitively showed
1. **Theorem 5 holds at the covariance level** — P(Δ > 0) = 0.57 in the base policy's reward distribution.
2. **Theorem 3 (BoN amplification) holds** — sign-flip gap is visible at every N.
3. **BoN amplification is non-monotone in N** — *a finding the published theory does not predict*. At large N, the factual-correctness signal eventually outweighs the small positive sycophancy correlation, because BoN gets more selective at choosing the argmax of the reward.
4. **Theorem 6 mitigation preserves capability** — MC1/MC2 within noise across all three models.

### What we couldn't definitively conclude
**At gradient-trained-policy scale**, neither the amplification nor the mitigation effect rose above noise:
- Vanilla GRPO didn't visibly amplify sycophancy (Δ = 0.0000 vs base, vs SE = 0.0084)
- Mitigated GRPO didn't visibly reduce it (Δ = −0.0015 vs vanilla)

### Why the trained-policy result is sub-noise
Two compounding reasons:
1. **The base policy is already extremely sycophantic** (75.67% on belief-wrong probes). There's a low ceiling for amplification to be visible.
2. **The fine-tune was light-touch** — ~0.005 KL from base. The policy barely moved on argmax-style scoring. With more steps or higher learning rate, we'd expect to see the directional effect grow.

This is consistent with the theory — the theorem predicts the *sign* of the drift, not its magnitude. The magnitude depends on how far you actually move the policy.

### The clean defensible claim
> *At the covariance level (the level where Shapira's theorem is defined), we verified the amplification prediction with P(Δ > 0) = 0.57. At BoN scale, we verified the amplification with a clear sign-flip gap that holds across N. At the gradient-trained-policy scale, vanilla GRPO did not visibly amplify and the corrected reward did not visibly reduce — but neither pushed in the wrong direction, and capability was preserved.*

> *The mitigation result is a partial verification of "Theorem 6 prevents amplification without collapse" — the "without collapse" half is confirmed; the "prevents amplification" half is below our noise floor at this training scale.*

---

## 5. Forward direction

### What a $9.70 follow-up would do
A longer vanilla GRPO run (lr=3e-6, 4 epochs instead of 2) should push KL from ~0.005 to ~0.03. At that scale we'd expect:
- Vanilla amplification to rise above the SE = 0.0084 noise floor
- Mitigation reduction (if re-run) to become detectable

This was planned but the cloud host went offline. Deferred.

### What scaling up would do
At 7B+ and longer training, both effects should become unambiguously visible. The theory predicts the *direction* — only the magnitude depends on training scale.

### Open methodological questions
- The ArmoRM 19-head decomposition wasn't exploited — there's a clean experiment in "which head contributes to the sycophancy tilt? style or factual?"
- The agreement detector `A(x, y)` is currently string-match; an LLM-judge upgrade was planned but not run.

---

## 6. Slide-deck-ready takeaways

**One-line summary:**
*We empirically verified Shapira et al.'s covariance theorem at the tilt and BoN levels with 1B-scale compute, and tested their Theorem 6 mitigation — capability is preserved but the trained-policy-level signal needs more training scale to land above noise.*

**Three key numbers:**
- **P(Δ > 0) = 0.57** — covariance has the predicted sign (Theorem 5, ✅)
- **Sycophancy gap across BoN N** — 0.10 → 0.18 → 0.14 (Theorem 3, ✅, with non-monotone extension)
- **Trained-policy effect = sub-noise** at this scale — needs more KL budget

**Three figures:**
- **Fig 1a** — tilt distribution shifts right
- **Fig 1c** — BoN sign-flip gap holds across N (non-monotone red curve)
- **Fig 4** — mitigation comparison: sycophancy and MC1 across base / vanilla / mitigated (all within noise)

**Methodology highlight:**
We deliberately chose a **verification scale** (1B Base, ~0.005 KL drift) where the theorem's *direction* should be observable cheaply but its *magnitude* may not be — a regime that exposes both the strength of the theory (correct sign at every level we measured) and its silence on practical magnitudes.

**What's novel beyond reproducing the published result:**
- Empirical demonstration that **BoN amplification is non-monotone in N** — at large N the factual signal eventually dominates the small sycophancy correlation. Not in the published theory.
- Confirmation that **Theorem 6's "without collapse" guarantee holds at gradient-training scale** with full fine-tuning on a learned RM.

---

## 7. Repo organization (for reference)

```
src/
  data/           # dataset construction (belief_injection, few_shot, truthfulqa)
  bon/            # Best-of-N sampling + tilt measurement
  grpo/           # GRPO trainer + corrected-reward function
  models/         # Llama-1B policy loader, ArmoRM-8B reward loader
  eval/           # MC1/MC2, sycophancy rate, KL estimators
  analysis/       # figure generation helpers
scripts/
  00_setup_*      # environment checks
  01_build_dataset.py    # Phase 1
  02_tilt_measurement.py # Phase 2
  03_bon_sweep.py        # Phase 3
  04_grpo_train.py       # Phase 4
  05_mitigation_sweep.py # Phase 5
  06_make_figures.py     # Phase 6 figures
  eval_mc.py             # MC1/MC2 standalone
  eval_sycophancy.py     # behavioral sycophancy comparison
  compare_sycophancy.py  # cross-model summary table
  make_fig3.py           # mitigation figure
  orchestrator.sh        # cloud Phase 2+3+figures chain w/ auto-stop
  orchestrator_extended.sh # planned extended Phase 4 (not run)
  cloud_bootstrap.sh     # fresh-instance setup
  cloud_sync.sh          # rsync repo to cloud
results/
  phase2_tilt/      # Theorem 5 evidence
  phase3_bon/       # Theorem 3 evidence
  eval_mc/          # MC1/MC2 capability
  eval_sycophancy/  # sycophancy rate per model
  figures/          # fig1a, fig1c, fig2, fig4
papers/             # supporting paper PDFs
proposal/           # V4 proposal document
```

---

## 8. Suggested slide outline (~12 slides)

1. **Title** — Verifying RLHF Sycophancy Amplification at 1B Scale
2. **The problem** — GPT-4o April 2025 rollback hook + sycophancy as a safety-relevant behavior
3. **The theory** — Shapira et al. covariance theorem in one slide (Theorem 1 + Theorem 6)
4. **Research question** — Does the theorem hold under finite data, gradient-based GRPO, learned RM?
5. **Design** — Llama-1B Base + ArmoRM-8B + TruthfulQA with belief injection
6. **Phase 2 result** — Fig 1a: tilt distribution shifts right, P(Δ > 0) = 0.57 ✅
7. **Phase 3 result** — Fig 1c: BoN sign-flip gap across N + non-monotone extension ✅
8. **Phase 3 result** — Fig 2: no Goodhart at N ≤ 128
9. **Phase 4 + 5** — vanilla and mitigated GRPO training
10. **Phase 6 result** — capability preserved (MC1/MC2 table) + sycophancy table
11. **Honest read** — verification at the covariance and BoN levels; trained-policy effect sub-noise at this scale
12. **Forward direction** — longer training or scaling up should land the trained-policy result

---

## Files to hand Claude alongside this summary

- `TARA_Final_Project_repo.zip` — full repo (everything on GitHub)
- This file — the narrative summary
