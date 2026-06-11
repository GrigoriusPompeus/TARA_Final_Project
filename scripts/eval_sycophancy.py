"""Sycophancy rate evaluation on belief-injected probes.

Scores every choice for each belief_wrong probe by log-prob under the policy,
takes the argmax, and classifies the response as A=1 if it matches the user's
stated (wrong) belief, or as correct if it matches the gold answer.

Usage:
    python -m scripts.eval_sycophancy                          # base model
    python -m scripts.eval_sycophancy --model_path checkpoints/grpo_vanilla/final
    python -m scripts.eval_sycophancy --n_items 100            # quick subset
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import CFG, ensure_dirs
from src.data.few_shot import build_prefix
from src.eval.sycophancy_rate import compute
from src.models.policy import batched_logprob_of_completions, load_policy


def load_belief_wrong(path: Path) -> list[dict]:
    probes = []
    with path.open() as f:
        for line in f:
            rec = json.loads(line)
            if rec["stance"] == "belief_wrong":
                probes.append(rec)
    return probes


def load_neutral_lookup(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with path.open() as f:
        for line in f:
            rec = json.loads(line)
            if rec["stance"] == "neutral":
                out[rec["source_id"]] = rec
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default=None,
                    help="HF repo id or local checkpoint path. Defaults to base.")
    ap.add_argument("--probes", default=str(CFG.paths.data_processed / "probes_all.jsonl"))
    ap.add_argument("--n_items", type=int, default=None,
                    help="If set, eval only the first N belief_wrong probes.")
    ap.add_argument("--no_few_shot", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--compute_flip", action="store_true",
                    help="Also score the matching neutral probes and report flip rate.")
    args = ap.parse_args()

    ensure_dirs()
    probes_path = Path(args.probes)
    belief_wrong = load_belief_wrong(probes_path)
    if args.n_items is not None:
        belief_wrong = belief_wrong[: args.n_items]
    print(f"Loaded {len(belief_wrong)} belief_wrong probes from {probes_path}")

    policy = load_policy(model_id=args.model_path)
    print(f"Policy loaded: device={policy.device} dtype={policy.dtype}")

    prefix = "" if args.no_few_shot else build_prefix()

    per_prompt: list[dict] = []
    for i, rec in enumerate(belief_wrong):
        prompt = f"{prefix}Q: {rec['text']}\nA: "
        logprobs = batched_logprob_of_completions(policy, prompt, rec["choices"])
        chosen = int(max(range(len(logprobs)), key=lambda k: logprobs[k]))
        per_prompt.append({
            "source_id": rec["source_id"],
            "chosen_idx": chosen,
            "user_stated_idx": rec["injected_idx"],
            "correct_idx": rec["correct_idx"],
            "logprobs": logprobs,
        })
        if (i + 1) % 100 == 0:
            print(f"  scored {i + 1}/{len(belief_wrong)}")

    neutral_lookup: dict[str, int] | None = None
    if args.compute_flip:
        print("Scoring neutral counterparts for flip-rate...")
        neutral_recs = load_neutral_lookup(probes_path)
        neutral_lookup = {}
        ids_needed = {r["source_id"] for r in per_prompt}
        for sid in ids_needed:
            if sid not in neutral_recs:
                continue
            nrec = neutral_recs[sid]
            prompt = f"{prefix}Q: {nrec['text']}\nA: "
            lps = batched_logprob_of_completions(policy, prompt, nrec["choices"])
            neutral_lookup[sid] = int(max(range(len(lps)), key=lambda k: lps[k]))

    result = compute(per_prompt, neutral_lookup=neutral_lookup)

    print()
    print(f"n_prompts         = {result.n_prompts}")
    print(f"sycophancy_rate   = {result.sycophancy_rate:.4f}")
    print(f"truthfulness_rate = {result.truthfulness_rate:.4f}")
    if result.flip_rate is not None:
        print(f"flip_rate         = {result.flip_rate:.4f}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "model_path": args.model_path or CFG.models.policy,
            "n_prompts": result.n_prompts,
            "sycophancy_rate": result.sycophancy_rate,
            "truthfulness_rate": result.truthfulness_rate,
            "flip_rate": result.flip_rate,
            "per_prompt": per_prompt,
        }, indent=2))
        print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
