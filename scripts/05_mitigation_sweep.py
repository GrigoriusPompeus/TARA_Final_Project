"""Phase 5 entry point: Shapira Theorem 6 mitigation lambda sweep.

Runs a series of GRPO training jobs at lambda ∈ {0, 0.5, 1.0, 2.0} and
compares each to the vanilla baseline.

Outputs:
    checkpoints/grpo_mitigated_lam{lam}/   per-lambda final model
    results/phase5_mitigation/summary.json

Note: this script invokes `scripts.04_grpo_train` as a subprocess for each
lambda so each run gets a fresh trainer + clean GPU state. It assumes we
are on a CUDA box with TRL installed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from src.config import CFG


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambdas", type=float, nargs="+", default=list(CFG.grpo.lambda_sweep))
    ap.add_argument("--n_train", type=int, default=1000)
    ap.add_argument("--n_epochs", type=int, default=1)
    ap.add_argument("--lora", action="store_true")
    args = ap.parse_args()

    out_dir = CFG.paths.results / "phase5_mitigation"
    out_dir.mkdir(parents=True, exist_ok=True)

    runs: list[dict] = []
    for lam in args.lambdas:
        name = f"grpo_lam{lam}".replace(".", "p")
        print(f"\n=== Running lambda={lam} as '{name}' ===")
        cmd = [
            sys.executable,
            "-m",
            "scripts.04_grpo_train",
            "--run_name",
            name,
            "--n_train",
            str(args.n_train),
            "--n_epochs",
            str(args.n_epochs),
            "--mitigation_lambda",
            str(lam),
        ]
        if args.lora:
            cmd.append("--lora")
        result = subprocess.run(cmd, check=False)
        runs.append({
            "lambda": lam,
            "run_name": name,
            "exit_code": result.returncode,
            "checkpoint": str(CFG.paths.checkpoints / name / "final"),
        })

    summary_path = out_dir / "runs.json"
    summary_path.write_text(json.dumps(runs, indent=2))
    print(f"\nWrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
