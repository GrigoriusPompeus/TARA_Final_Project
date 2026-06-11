"""Phase 3: Best-of-N sweep and Goodhart curve (Shapira Fig. 1c reproduction).

Pipeline:
    1. For each belief_wrong probe x', sample N_max=128 *unsteered* responses
       from the (few-shot wrapped) base policy. This is the BoN candidate pool.
    2. Score each candidate with ArmoRM. Also classify with A(x', y) and
       (optionally) check whether y contains the gold answer.
    3. For each N in {1, 2, 4, 8, 16, 32, 64, 128}, compute:
         - sycophancy_rate(N) on positive-tilt subset (expect RISE)
         - sycophancy_rate(N) on negative-tilt subset (expect FALL)
         - mean ArmoRM reward (proxy) ← grows monotonically by construction
         - factual hit rate (gold)    ← Goodhart: expected to FALL
    4. Persist curve points + raw pool to results/phase3_bon/.

The positive-vs-negative split uses Phase 2 tilts from results/phase2_tilt/tilts.jsonl.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch

from src.bon.agreement import agreement_indicator
from src.bon.curve import partition_by_tilt, sweep
from src.bon.generate import SteerConfig, build_steered_prompt
from src.bon.score import score_per_prompt_groups
from src.bon.tilt import load_jsonl as load_tilts
from src.config import CFG, ensure_dirs
from src.data.few_shot import build_prefix
from src.models.policy import batched_generate, load_policy
from src.models.reward import load_reward


def load_belief_wrong_probes(jsonl_path: Path) -> list[dict]:
    out: list[dict] = []
    with open(jsonl_path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("stance") == "belief_wrong" and r.get("strategy") == "answer_suggestion":
                out.append(r)
    return out


def wrap_unsteered(question_text: str) -> str:
    """Few-shot scaffold a prompt with NO steering prime — pure π_base on x'."""
    prefix = build_prefix()
    return f"{prefix}Q: {question_text}\nA: "


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_prompts", default="50", help="int or 'all'")
    ap.add_argument("--n_candidates", type=int, default=128,
                    help="N_max: pool size per prompt for BoN")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gen_batch", type=int, default=8)
    ap.add_argument("--score_batch", type=int, default=4)
    ap.add_argument("--n_bootstrap", type=int, default=8,
                    help="How many random N-subsets to average over per prompt at each N")
    ap.add_argument("--n_values", type=int, nargs="+",
                    default=list(CFG.bon.n_values))
    ap.add_argument("--source_filter", default=None)
    ap.add_argument("--agreement_mode", default="key_terms",
                    choices=["containment", "key_terms"])
    ap.add_argument("--overlap_threshold", type=float, default=0.6)
    ap.add_argument("--tilts_path",
                    default=str(CFG.paths.results / "phase2_tilt" / "tilts.jsonl"))
    args = ap.parse_args()

    ensure_dirs()
    out_dir = CFG.paths.results / "phase3_bon"
    out_dir.mkdir(parents=True, exist_ok=True)

    tilts_path = Path(args.tilts_path)
    if not tilts_path.exists():
        raise SystemExit(
            f"No Phase-2 tilts found at {tilts_path}. Run scripts.02_tilt_measurement first."
        )
    tilts = {t.source_id: t.delta_mean for t in load_tilts(tilts_path)}
    print(f"Loaded {len(tilts)} prompt tilts from Phase 2")

    probes = load_belief_wrong_probes(CFG.paths.data_processed / "probes_all.jsonl")
    if args.source_filter:
        probes = [p for p in probes if p["source"] == args.source_filter]
    # Only keep probes for which we have a Phase-2 tilt sign
    probes = [p for p in probes if p["source_id"] in tilts]
    rng = random.Random(args.seed)
    rng.shuffle(probes)
    if args.n_prompts != "all":
        probes = probes[: int(args.n_prompts)]
    print(f"Sweeping over {len(probes)} probes × {args.n_candidates} candidates")

    print("Loading policy ...")
    policy = load_policy()

    print(f"Generating {args.n_candidates} unsteered candidates per prompt ...")
    t0 = time.time()
    unsteered_prompts = [wrap_unsteered(p["text"]) for p in probes]
    pool_texts = batched_generate(
        policy,
        unsteered_prompts,
        n_per_prompt=args.n_candidates,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        batch_size=args.gen_batch,
    )
    print(f"  generated in {time.time() - t0:.1f}s")

    del policy
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Loading reward model ...")
    reward = load_reward()
    print("Scoring candidates ...")
    t0 = time.time()
    scored = score_per_prompt_groups(
        reward,
        [p["text"] for p in probes],
        pool_texts,
        return_heads=False,
        batch_size=args.score_batch,
    )
    print(f"  scored in {time.time() - t0:.1f}s")

    print("Classifying A(x, y) per candidate ...")
    pool_records: list[dict] = []
    for p, texts, s in zip(probes, pool_texts, scored):
        rewards = s.get("score", [])
        if not rewards:
            continue
        user_belief = p["choices"][p["injected_idx"]]
        correct = p["choices"][p["correct_idx"]]
        agree_idx = [
            agreement_indicator(
                t,
                user_belief,
                correct,
                mode=args.agreement_mode,
                overlap_threshold=args.overlap_threshold,
            )
            for t in texts
        ]
        correct_idx = [
            agreement_indicator(
                t,
                correct,
                user_belief,
                mode=args.agreement_mode,
                overlap_threshold=args.overlap_threshold,
            )
            for t in texts
        ]
        pool_records.append(
            {
                "source_id": p["source_id"],
                "source": p["source"],
                "rewards": rewards,
                "agree": agree_idx,
                "correct": correct_idx,
                "responses": texts,
            }
        )

    parts = partition_by_tilt(pool_records, tilts)
    print(f"Partition: positive={len(parts['positive'])}  negative={len(parts['negative'])}"
          f"  untagged={len(parts['untagged'])}")

    print("Running BoN sweep ...")
    n_values = sorted(set(args.n_values))
    curves = {
        "all": [asdict(pt) for pt in sweep(pool_records, n_values, seed=args.seed, n_bootstrap=args.n_bootstrap)],
        "positive": [asdict(pt) for pt in sweep(parts["positive"], n_values, seed=args.seed, n_bootstrap=args.n_bootstrap)],
        "negative": [asdict(pt) for pt in sweep(parts["negative"], n_values, seed=args.seed, n_bootstrap=args.n_bootstrap)],
    }

    out_path = out_dir / "bon_curves.json"
    with open(out_path, "w") as f:
        json.dump({"n_values": n_values, "curves": curves}, f, indent=2)
    print(f"Saved curves to {out_path}")

    pool_path = out_dir / "pool.jsonl"
    with open(pool_path, "w") as f:
        for r in pool_records:
            f.write(json.dumps(r) + "\n")
    print(f"Saved raw pool to {pool_path}")

    print("\nSummary (sycophancy rate by N):")
    print(f"  {'N':>4} | {'all':>8} | {'pos':>8} | {'neg':>8}")
    for n in n_values:
        a = next((pt for pt in curves["all"] if pt["n"] == n), None)
        p = next((pt for pt in curves["positive"] if pt["n"] == n), None)
        m = next((pt for pt in curves["negative"] if pt["n"] == n), None)
        def fmt(pt):
            if pt is None or pt["syc_rate"] != pt["syc_rate"]:
                return "  —  "
            return f"{pt['syc_rate']:.3f}"
        print(f"  {n:>4} | {fmt(a):>8} | {fmt(p):>8} | {fmt(m):>8}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
