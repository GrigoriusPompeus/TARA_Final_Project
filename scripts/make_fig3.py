"""Fig 3: Shapira Theorem 6 mitigation — sycophancy ↓, capability preserved.

Reads:
  results/eval_sycophancy/{base,grpo_vanilla,grpo_mitigated_lam1}.json
  results/eval_mc/{base,grpo_vanilla,grpo_mitigated_lam1}.json
Writes:
  results/figures/fig3_mitigation_comparison.{png,pdf}
"""

from __future__ import annotations

import json

from src.analysis.figures import fig_mitigation_comparison
from src.config import CFG, ensure_dirs


def _load(path):
    return json.loads(path.read_text())


def main() -> int:
    ensure_dirs()
    syc_dir = CFG.paths.results / "eval_sycophancy"
    mc_dir = CFG.paths.results / "eval_mc"

    base_syc = _load(syc_dir / "base.json")["sycophancy_rate"]
    van_syc  = _load(syc_dir / "grpo_vanilla.json")["sycophancy_rate"]
    mit_syc  = _load(syc_dir / "grpo_mitigated_lam1.json")["sycophancy_rate"]

    base_mc1 = _load(mc_dir / "base.json")["mc1_mean"]
    van_mc1  = _load(mc_dir / "grpo_vanilla.json")["mc1_mean"]
    mit_mc1  = _load(mc_dir / "grpo_mitigated_lam1.json")["mc1_mean"]

    # x-axis: λ values (vanilla GRPO = λ=0, mitigated = λ=1.0)
    lambdas = [0.0, 1.0]
    syc_rates = [van_syc, mit_syc]
    mc1s      = [van_mc1, mit_mc1]

    fig_mitigation_comparison(
        lambdas=lambdas,
        syc_rates=syc_rates,
        mc1s=mc1s,
        baseline_syc=base_syc,
        baseline_mc1=base_mc1,
        out_dir=CFG.paths.results / "figures",
    )
    print("Wrote fig3_mitigation_comparison.{png,pdf}")
    print(f"  base_syc={base_syc:.4f}  vanilla_syc={van_syc:.4f}  mitigated_syc={mit_syc:.4f}")
    print(f"  base_mc1={base_mc1:.4f}  vanilla_mc1={van_mc1:.4f}  mitigated_mc1={mit_mc1:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
