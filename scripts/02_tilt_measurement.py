"""Phase 2: Reward-tilt measurement (Shapira Fig. 1a/1b reproduction).

Pipeline:
    1. Load belief_wrong probes from data/processed/probes_all.jsonl.
    2. Optionally subsample for the local pilot.
    3. For each prompt: generate n_agree agreeing + n_correct correcting
       responses from the few-shot scaffolded base policy.
    4. Score all responses with ArmoRM (gated MoE score + 19 per-objective heads).
    5. Compute Δ̂_mean(x') and per-head Δ̂ per prompt.
    6. Aggregate: sycophancy rate = P_x(Δ̂_mean > 0).
    7. Persist per-sample rewards (needed for Phase 3 BoN sweep) and per-prompt
       tilts under results/phase2_tilt/.

CLI:
    python -m scripts.02_tilt_measurement \
        --n_prompts 50 --n_per_group 16          # local smoke test
    python -m scripts.02_tilt_measurement \
        --n_prompts all --n_per_group 64         # cloud full run
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch

from src.bon.generate import generate_paired
from src.bon.score import score_per_prompt_groups
from src.bon.tilt import compute_one, save_jsonl, sycophancy_rate
from src.config import CFG, ensure_dirs
from src.models.policy import load_policy
from src.models.reward import load_reward


def load_belief_wrong_probes(jsonl_path: Path) -> list[dict]:
    out: list[dict] = []
    with open(jsonl_path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("stance") == "belief_wrong" and r.get("strategy") == "answer_suggestion":
                out.append(r)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_prompts", default="50", help="int or 'all'")
    ap.add_argument("--n_per_group", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gen_batch", type=int, default=8)
    ap.add_argument("--score_batch", type=int, default=4)
    ap.add_argument("--save_responses", action="store_true",
                    help="Also persist raw response strings (large files).")
    ap.add_argument("--source_filter", default=None,
                    help="Only use probes whose source matches this string (e.g. 'truthfulqa').")
    ap.add_argument("--no_heads", action="store_true",
                    help="Skip ArmoRM per-head decomposition (faster, less analysis).")
    args = ap.parse_args()

    ensure_dirs()
    out_dir = CFG.paths.results / "phase2_tilt"
    out_dir.mkdir(parents=True, exist_ok=True)

    probes_path = CFG.paths.data_processed / "probes_all.jsonl"
    print(f"Loading belief_wrong probes from {probes_path} ...")
    probes = load_belief_wrong_probes(probes_path)
    print(f"  {len(probes)} total belief_wrong probes")

    if args.source_filter:
        probes = [p for p in probes if p["source"] == args.source_filter]
        print(f"  {len(probes)} after source filter '{args.source_filter}'")

    rng = random.Random(args.seed)
    rng.shuffle(probes)
    if args.n_prompts != "all":
        n = int(args.n_prompts)
        probes = probes[:n]
    print(f"  using {len(probes)} probes")

    print("Loading policy ...")
    policy = load_policy()
    print(f"  policy on {policy.device} ({policy.dtype})")

    print("Generating agreeing/correcting groups ...")
    t0 = time.time()
    question_texts = [p["text"] for p in probes]
    groups = generate_paired(
        policy,
        question_texts,
        n_agreeing=args.n_per_group,
        n_correcting=args.n_per_group,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        batch_size=args.gen_batch,
    )
    print(f"  generated in {time.time() - t0:.1f}s")

    # Free policy from memory before loading reward (matters on M1's unified mem)
    del policy
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Loading reward model ...")
    reward = load_reward()
    print(f"  reward on {reward.device} ({reward.dtype})")

    print("Scoring agreeing group ...")
    t0 = time.time()
    agree_scores = score_per_prompt_groups(
        reward,
        question_texts,
        groups["agree"],
        return_heads=not args.no_heads,
        batch_size=args.score_batch,
    )
    print(f"  done in {time.time() - t0:.1f}s")

    print("Scoring correcting group ...")
    t0 = time.time()
    correct_scores = score_per_prompt_groups(
        reward,
        question_texts,
        groups["correct"],
        return_heads=not args.no_heads,
        batch_size=args.score_batch,
    )
    print(f"  done in {time.time() - t0:.1f}s")

    print("Computing per-prompt tilts ...")
    tilts = []
    for p, ag, co in zip(probes, agree_scores, correct_scores):
        tilts.append(
            compute_one(
                source_id=p["source_id"],
                source=p["source"],
                scores_agree=ag.get("score", []),
                scores_correct=co.get("score", []),
                heads_agree=ag.get("heads") if not args.no_heads else None,
                heads_correct=co.get("heads") if not args.no_heads else None,
            )
        )

    syc_rate = sycophancy_rate(tilts)
    delta_means = [t.delta_mean for t in tilts if t.delta_mean == t.delta_mean]
    print(f"\nSycophancy rate (Δ̂_mean > 0): {syc_rate:.3f}  ({len(delta_means)} valid prompts)")
    if delta_means:
        delta_means.sort()
        print(f"Δ̂_mean: min={delta_means[0]:.3f}  median={delta_means[len(delta_means)//2]:.3f}  max={delta_means[-1]:.3f}")

    tilts_path = out_dir / "tilts.jsonl"
    save_jsonl(tilts, tilts_path)
    print(f"\nSaved per-prompt tilts to {tilts_path}")

    # Persist samples (for the Phase 3 BoN sweep we will need to re-score, but
    # if --save_responses is set we cache the raw strings to skip re-generation).
    if args.save_responses:
        responses_path = out_dir / "responses.jsonl"
        with open(responses_path, "w") as f:
            for p, ag_resp, co_resp, ag_sc, co_sc in zip(
                probes, groups["agree"], groups["correct"], agree_scores, correct_scores
            ):
                row = {
                    "source_id": p["source_id"],
                    "source": p["source"],
                    "prompt": p["text"],
                    "agree": [
                        {"text": t, "score": s}
                        for t, s in zip(ag_resp, ag_sc.get("score", []))
                    ],
                    "correct": [
                        {"text": t, "score": s}
                        for t, s in zip(co_resp, co_sc.get("score", []))
                    ],
                }
                f.write(json.dumps(row) + "\n")
        print(f"Saved raw responses to {responses_path}")

    summary = {
        "n_prompts": len(probes),
        "n_per_group": args.n_per_group,
        "sycophancy_rate": syc_rate,
        "delta_mean_quantiles": {
            "min": delta_means[0] if delta_means else None,
            "p25": delta_means[len(delta_means) // 4] if delta_means else None,
            "median": delta_means[len(delta_means) // 2] if delta_means else None,
            "p75": delta_means[(3 * len(delta_means)) // 4] if delta_means else None,
            "max": delta_means[-1] if delta_means else None,
        },
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
