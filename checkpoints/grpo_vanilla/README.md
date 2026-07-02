---
base_model: meta-llama/Llama-3.2-1B
library_name: transformers
model_name: grpo_vanilla
tags:
  - generated_from_trainer
  - grpo
  - trl
license: llama3.2
---

# grpo_vanilla

Vanilla GRPO fine-tune of [meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B)
used as the untreated baseline in the TARA sycophancy verification project.

- **Reward model:** [RLHFlow/ArmoRM-Llama3-8B-v0.1](https://huggingface.co/RLHFlow/ArmoRM-Llama3-8B-v0.1)
- **Algorithm:** GRPO ([Shao et al. 2024](https://arxiv.org/abs/2402.03300)), TRL's `GRPOTrainer`.
- **Hyperparameters:** G=8, β=0.04, lr=1e-6, n_epochs=2, full fine-tuning on ~900 belief-injected TruthfulQA + sycophancy_eval probes.
- **Hardware:** A100 80GB.

`model.safetensors` is not checked into git (≈2.5 GB). To recreate, see the
project root's `CHECKPOINTS.md`.

## Framework versions

- TRL 1.4.0
- Transformers 4.57.6
- PyTorch 2.11.0 + CUDA 12.8
- Datasets 4.8.5
- Tokenizers 0.22.2
