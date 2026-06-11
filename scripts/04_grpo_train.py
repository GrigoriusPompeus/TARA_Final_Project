"""Phase 4 entry point: GRPO training.

Examples:
    # Cloud A100, vanilla GRPO with KL penalty 0.04, 1000 prompts, 2 epochs
    python -m scripts.04_grpo_train --run_name grpo_vanilla

    # LoRA + small training set for a cheap pilot
    python -m scripts.04_grpo_train --lora --n_train 300 --n_epochs 1 \
        --run_name grpo_pilot_lora

    # Mitigation: Shapira Theorem 6 with lambda=1.0
    python -m scripts.04_grpo_train --mitigation_lambda 1.0 \
        --run_name grpo_mitigated_lam1
"""

from src.grpo.train import main

if __name__ == "__main__":
    raise SystemExit(main())
