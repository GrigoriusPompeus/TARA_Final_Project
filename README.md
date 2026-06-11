# TARA Final Project — Verifying RLHF Sycophancy Amplification

Empirical verification of Shapira, Benade, Procaccia (Feb 2026, [arXiv:2602.01002](https://arxiv.org/abs/2602.01002)) "How RLHF Amplifies Sycophancy" at 1B scale.

## Project structure

```
src/
  data/       belief-injection probe dataset construction
  models/     Llama-3.2-1B Base policy + ArmoRM-8B reward loaders
  eval/       TruthfulQA MC1/MC2, sycophancy rate, Beirami KL estimators
  bon/        Best-of-N: candidate generation, scoring, tilt, BoN curves
  grpo/       TRL GRPO training + Shapira Theorem 6 corrected reward
  analysis/   figures and stats
configs/      yaml hyperparameter configs
scripts/      thin entry points 00_setup.sh, 01..06_*.py
data/         raw + processed datasets (gitignored)
results/      experiment outputs (gitignored)
papers/       supporting paper PDFs
proposal/     V4 proposal
feedback/     reviewer feedback PDFs
```

## Phases

| Phase | Where  | Cost  | What                                                 |
| ----- | ------ | ----- | ---------------------------------------------------- |
| 0     | local  | $0    | env + dataset download + smoke test                  |
| 1     | local  | $0    | belief-inject TruthfulQA + SycophancyEval            |
| 2     | local* | $0–3  | reward-tilt measurement (Shapira Fig. 1a/1b)         |
| 3     | local* | $0–5  | BoN sweep + Goodhart curve (Shapira Fig. 1c)         |
| 4     | cloud  | $15–25 | GRPO training on Vast.ai A100 80GB                   |
| 5     | cloud  | $10–15 | Theorem 6 KL-penalty mitigation, λ sweep             |
| 6     | local  | $0    | analysis, figures, write-up                          |

\* Phases 2-3 default to M1 Max local; a 1-hour cloud burst is the fallback if generation throughput is the blocker.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[local,dev]"          # M1 Mac
# OR
pip install -e ".[cloud,dev]"          # Linux+CUDA box
```

You'll also need:
- A HuggingFace account that has accepted the Llama-3.2 license at https://huggingface.co/meta-llama/Llama-3.2-1B
- `huggingface-cli login` with a token that has `read` access to gated repos.

## Citations (verified)

- Shapira, Benade, Procaccia (Feb 2026). *How RLHF Amplifies Sycophancy.* arXiv:2602.01002.
- Beirami et al. (ICML 2025). *Theoretical Guarantees on the Best-of-n Alignment Policy.* arXiv:2401.01879.
- Shao et al. (2024). *DeepSeekMath: Pushing the Limits of Mathematical Reasoning.* arXiv:2402.03300. (Origin of GRPO.)
- Sharma et al. (ICLR 2024). *Towards Understanding Sycophancy in Language Models.* arXiv:2310.13548.
- Wang et al. (EMNLP 2024). *Interpretable Preferences via Multi-Objective Reward Modeling and Mixture-of-Experts (ArmoRM).* arXiv:2406.12845.
- Lin, Hilton, Evans (2022). *TruthfulQA: Measuring How Models Mimic Human Falsehoods.* arXiv:2109.07958.
- Perez et al. (2022). *Discovering Language Model Behaviors with Model-Written Evaluations.* arXiv:2212.09251.
- OpenAI (29 Apr 2025). *Sycophancy in GPT-4o: what happened and what we're doing about it.*
- OpenAI (~2 May 2025). *Expanding on what we missed with sycophancy.*
