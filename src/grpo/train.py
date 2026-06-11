"""GRPO training entry point.

Uses TRL's GRPOTrainer (Shao et al. 2024 algorithm; recent TRL versions
expose this as a clean Trainer class). Designed for a single A100 80GB:
    - policy: Llama-3.2-1B Base (~2 GB)
    - reference policy: another copy (~2 GB; managed by GRPOTrainer)
    - reward model: ArmoRM-Llama3-8B in bf16 (~16 GB)
    - rollouts + AdamW state: ~10 GB
    Total ≈ 32 GB; fits comfortably.

The KL penalty β > 0 stays ON (Feedback (1) §B-critical). The group size G
defaults to 8 (Shao et al. used G=64 for 7B; G=8 is appropriate for 1B
within memory).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import CFG, ensure_dirs
from src.grpo.dataset import load_probes, to_hf_dataset, train_eval_split
from src.grpo.reward_fn import armorm_reward, make_corrected_reward


def _import_trl():
    # Import lazily so the module is importable on machines without TRL
    # (e.g., local M1 box). Cloud A100 will have it.
    try:
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as e:
        raise SystemExit(
            "TRL is required for GRPO training. Install with "
            "`pip install -e .[cloud]` on a CUDA box."
        ) from e
    return GRPOConfig, GRPOTrainer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=1000,
                    help="Number of belief_wrong + belief_correct + neutral probes total")
    ap.add_argument("--group_size", type=int, default=CFG.grpo.group_size)
    ap.add_argument("--kl_beta", type=float, default=CFG.grpo.kl_beta)
    ap.add_argument("--learning_rate", type=float, default=CFG.grpo.learning_rate)
    ap.add_argument("--batch_size", type=int, default=CFG.grpo.batch_size)
    ap.add_argument("--n_epochs", type=int, default=CFG.grpo.n_epochs)
    ap.add_argument("--max_prompt_length", type=int, default=CFG.grpo.max_prompt_length)
    ap.add_argument("--max_new_tokens", type=int, default=CFG.grpo.max_new_tokens)
    ap.add_argument("--seed", type=int, default=CFG.grpo.seed)
    ap.add_argument("--mitigation_lambda", type=float, default=0.0,
                    help="Shapira Theorem 6 lambda. 0 = vanilla GRPO; >0 = corrected reward.")
    ap.add_argument("--run_name", default="grpo")
    ap.add_argument("--save_every", type=int, default=CFG.grpo.save_every_steps)
    ap.add_argument("--lora", action="store_true",
                    help="Train with LoRA (PEFT) instead of full fine-tuning.")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    args = ap.parse_args()

    ensure_dirs()
    GRPOConfig, GRPOTrainer = _import_trl()

    out_dir = CFG.paths.checkpoints / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = CFG.paths.logs / args.run_name
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading policy {CFG.models.policy} ...")
    tok = AutoTokenizer.from_pretrained(CFG.models.policy)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        CFG.models.policy,
        torch_dtype=torch.bfloat16,
    )

    if args.lora:
        from peft import LoraConfig

        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
    else:
        peft_config = None

    n_per = max(1, args.n_train // 3)
    rows = load_probes(n_per_stratum=n_per, seed=args.seed)
    train_rows, eval_rows = train_eval_split(rows, eval_frac=0.1, seed=args.seed)
    print(f"Train: {len(train_rows)}  Eval: {len(eval_rows)}")
    train_ds = to_hf_dataset(train_rows)
    eval_ds = to_hf_dataset(eval_rows)

    if args.mitigation_lambda > 0:
        reward_fn = make_corrected_reward(lam=args.mitigation_lambda)
    else:
        reward_fn = armorm_reward

    grpo_config = GRPOConfig(
        output_dir=str(out_dir),
        run_name=args.run_name,
        num_generations=args.group_size,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        learning_rate=args.learning_rate,
        num_train_epochs=args.n_epochs,
        max_completion_length=args.max_new_tokens,
        beta=args.kl_beta,
        save_steps=args.save_every,
        eval_strategy="steps",
        eval_steps=CFG.grpo.eval_every_steps,
        logging_steps=10,
        bf16=True,
        seed=args.seed,
        report_to=[],
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        reward_funcs=reward_fn,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tok,
        peft_config=peft_config,
    )

    print("Starting GRPO training ...")
    trainer.train()
    trainer.save_model(str(out_dir / "final"))
    tok.save_pretrained(str(out_dir / "final"))

    summary = {
        "run_name": args.run_name,
        "n_train": len(train_rows),
        "n_eval": len(eval_rows),
        "group_size": args.group_size,
        "kl_beta": args.kl_beta,
        "mitigation_lambda": args.mitigation_lambda,
        "epochs": args.n_epochs,
        "lora": args.lora,
    }
    (log_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Done. Checkpoints at {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
