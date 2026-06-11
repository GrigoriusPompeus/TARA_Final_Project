"""Print side-by-side sycophancy / truthfulness comparison across the three policies.

Reads results/eval_sycophancy/{base,grpo_vanilla,grpo_mitigated_lam1}.json
and prints a table plus the predicted-ordering check:

    syc_rate(mitigated) < syc_rate(vanilla) <= syc_rate(base)   ← Shapira Theorem 6
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from src.config import CFG


MODELS = [
    ("Base Llama-3.2-1B",        "base.json"),
    ("Phase 4 vanilla GRPO λ=0", "grpo_vanilla.json"),
    ("Phase 5 mitigated λ=1.0",  "grpo_mitigated_lam1.json"),
]


def se_proportion(p: float, n: int) -> float:
    if n == 0:
        return 0.0
    return math.sqrt(p * (1.0 - p) / n)


def main() -> int:
    eval_dir = CFG.paths.results / "eval_sycophancy"
    print(f"Reading from {eval_dir}")
    print()
    rows = []
    for label, fname in MODELS:
        p = eval_dir / fname
        if not p.exists():
            print(f"  MISSING: {p}")
            continue
        d = json.loads(p.read_text())
        syc = d["sycophancy_rate"]
        tru = d["truthfulness_rate"]
        n = d["n_prompts"]
        rows.append((label, n, syc, tru, d.get("flip_rate")))

    if not rows:
        print("No results found.")
        return 1

    name_w = max(len(r[0]) for r in rows)
    print(f"{'Model'.ljust(name_w)}    n     syc_rate (±SE)     truth_rate (±SE)    flip")
    print("-" * (name_w + 64))
    for label, n, syc, tru, flip in rows:
        se_s = se_proportion(syc, n)
        se_t = se_proportion(tru, n)
        flip_s = f"{flip:.4f}" if flip is not None else "  -  "
        print(f"{label.ljust(name_w)}  {n:>4}    {syc:.4f} ± {se_s:.4f}    {tru:.4f} ± {se_t:.4f}    {flip_s}")
    print()

    by_label = {r[0]: r[2] for r in rows}
    base = by_label.get("Base Llama-3.2-1B")
    van  = by_label.get("Phase 4 vanilla GRPO λ=0")
    mit  = by_label.get("Phase 5 mitigated λ=1.0")
    if None not in (base, van, mit):
        amplified = van > base
        mitigated = mit < van
        print(f"Vanilla amplified sycophancy vs base?  {amplified}   (Δ = {van - base:+.4f})")
        print(f"Mitigation reduced sycophancy vs vanilla?  {mitigated}   (Δ = {mit - van:+.4f})")
        print()
        if amplified and mitigated:
            print("VERIFIED: ordering matches Shapira Theorem 5 (amplification) + Theorem 6 (mitigation).")
        elif mitigated and not amplified:
            print("PARTIAL: mitigation works, but vanilla GRPO didn't visibly amplify sycophancy.")
        elif amplified and not mitigated:
            print("PARTIAL: vanilla amplified, but mitigation didn't reduce — λ=1 too weak or detector mismatch.")
        else:
            print("NULL: neither effect detected at this scale.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
