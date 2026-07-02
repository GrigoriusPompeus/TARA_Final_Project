# Supporting papers

PDFs of these papers are not tracked in the repo (copyright). Read them at the
arXiv / ACL links below.

## The paper this project verifies

- **Shapira, Benade, Procaccia (Feb 2026).** *How RLHF Amplifies Sycophancy.*
  arXiv:2602.01002 — https://arxiv.org/abs/2602.01002
  The covariance theorem, Best-of-N amplification (Theorem 3), and the
  Theorem-6 mitigation this project verifies.

## Methods and tools we use

- **Shao et al. (2024).** *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.*
  arXiv:2402.03300 — https://arxiv.org/abs/2402.03300
  Original GRPO algorithm, implemented in TRL's `GRPOTrainer`.

- **Wang et al. (EMNLP 2024).** *Interpretable Preferences via Multi-Objective Reward Modeling and Mixture-of-Experts (ArmoRM).*
  arXiv:2406.12845 — https://arxiv.org/abs/2406.12845
  The reward model we use (`RLHFlow/ArmoRM-Llama3-8B-v0.1`), 19 interpretable heads.

- **Beirami et al. (ICML 2025).** *Theoretical Guarantees on the Best-of-n Alignment Policy.*
  arXiv:2401.01879 — https://arxiv.org/abs/2401.01879
  The KL bound (Eq. 25) used in the Phase 3 BoN sweep.

## Empirical evidence for sycophancy

- **Sharma et al. (ICLR 2024).** *Towards Understanding Sycophancy in Language Models.*
  arXiv:2310.13548 — https://arxiv.org/abs/2310.13548
  Probe-template inspiration; also documents sycophantic behaviour empirically.

- **Perez et al. (2022).** *Discovering Language Model Behaviors with Model-Written Evaluations.*
  arXiv:2212.09251 — https://arxiv.org/abs/2212.09251
  Broader landscape of RLHF-induced behavioural failures.

## Evaluation

- **Lin, Hilton, Evans (2022).** *TruthfulQA: Measuring How Models Mimic Human Falsehoods.*
  arXiv:2109.07958 — https://arxiv.org/abs/2109.07958
  MC1 / MC2 log-probability evaluation protocol.

## Industry context

- **OpenAI (29 Apr 2025).** *Sycophancy in GPT-4o: what happened and what we're doing about it.*
- **OpenAI (~2 May 2025).** *Expanding on what we missed with sycophancy.*
  Real-world example that motivates the project.
