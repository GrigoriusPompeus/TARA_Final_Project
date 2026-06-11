# Trained checkpoints

The GRPO-trained policy weights are not stored in this repo — each `model.safetensors`
is ~2.5 GB, well past GitHub's 100 MB per-file limit.

## How to recreate

Both runs use Llama-3.2-1B Base as the policy and ArmoRM-Llama3-8B-v0.1 as the
reward model. Full fine-tuning, ~1 epoch on ~1k prompts, on an 80 GB A100.

```bash
# Phase 4: vanilla GRPO baseline (λ = 0)
python -m scripts.04_grpo_train \
    --lambda_val 0.0 \
    --out checkpoints/grpo_vanilla

# Phase 5: Theorem-6 mitigated GRPO (λ = 1.0)
python -m scripts.05_mitigation_sweep \
    --lambda_val 1.0 \
    --out checkpoints/grpo_mitigated_lam1
```

Approximate wall-clock on a single A100 80 GB:

| Run | Time |
|---|---|
| Phase 4 vanilla | ~45 min |
| Phase 5 mitigated (λ=1.0) | ~105 min |

The corrected-reward term in Phase 5 adds per-scoring overhead, which is why the
mitigated run is ~2× slower.
