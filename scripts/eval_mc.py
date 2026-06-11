"""Standalone TruthfulQA MC1/MC2 evaluation.

Usage:
    python -m scripts.eval_mc                          # base model
    python -m scripts.eval_mc --model_path checkpoints/grpo_vanilla/final
    python -m scripts.eval_mc --n_items 100            # quick subset
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import CFG, ensure_dirs
from src.eval.run_mc import run
from src.models.policy import load_policy


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default=None,
                    help="HF repo id or local checkpoint path. Defaults to base.")
    ap.add_argument("--n_items", type=int, default=None)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--no_few_shot", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ensure_dirs()
    policy = load_policy(model_id=args.model_path)
    r = run(
        policy,
        split=args.split,
        n_items=args.n_items,
        use_few_shot_prefix=not args.no_few_shot,
    )

    print(f"MC1: {r.mc1_mean:.4f}   MC2: {r.mc2_mean:.4f}   n={r.n}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({
            "model_path": args.model_path or CFG.models.policy,
            "n_items": r.n,
            "mc1_mean": r.mc1_mean,
            "mc2_mean": r.mc2_mean,
        }, indent=2))
        print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
