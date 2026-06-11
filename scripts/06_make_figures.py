"""Phase 6: produce all canonical figures from on-disk results."""

from __future__ import annotations

import json
from pathlib import Path

from src.analysis.figures import (
    fig_bon_sign_flip,
    fig_goodhart_curve,
    fig_grpo_trajectory,
    fig_mitigation_comparison,
    fig_tilt_distribution,
)
from src.bon.tilt import load_jsonl as load_tilts
from src.config import CFG, ensure_dirs


def main() -> int:
    ensure_dirs()
    out_dir = CFG.paths.results / "figures"

    tilts_path = CFG.paths.results / "phase2_tilt" / "tilts.jsonl"
    if tilts_path.exists():
        tilts = load_tilts(tilts_path)
        fig_tilt_distribution([t.delta_mean for t in tilts], out_dir)
        print(f"Wrote fig 1a")

    bon_path = CFG.paths.results / "phase3_bon" / "bon_curves.json"
    if bon_path.exists():
        data = json.loads(bon_path.read_text())
        n_values = data["n_values"]
        pos = [pt["syc_rate"] for pt in data["curves"]["positive"]]
        neg = [pt["syc_rate"] for pt in data["curves"]["negative"]]
        allc = [pt["syc_rate"] for pt in data["curves"]["all"]]
        fig_bon_sign_flip(n_values, pos, neg, allc, out_dir)
        proxy = [pt["mean_reward"] for pt in data["curves"]["all"]]
        gold = [pt["correct_rate"] or 0.0 for pt in data["curves"]["all"]]
        fig_goodhart_curve(n_values, proxy, gold, out_dir)
        print("Wrote fig 1c + fig 2")

    trajectory_path = CFG.paths.logs / "grpo_vanilla" / "trajectory.jsonl"
    if trajectory_path.exists():
        steps, proxy, syc, mc1 = [], [], [], []
        with open(trajectory_path) as f:
            for line in f:
                r = json.loads(line)
                steps.append(r["step"])
                proxy.append(r.get("proxy_reward", 0.0))
                syc.append(r.get("sycophancy_rate", 0.0))
                mc1.append(r.get("mc1", 0.0))
        if steps:
            fig_grpo_trajectory(steps, proxy, syc, mc1, out_dir)
            print("Wrote fig 3")

    mitigation_path = CFG.paths.results / "phase5_mitigation" / "summary.json"
    if mitigation_path.exists():
        data = json.loads(mitigation_path.read_text())
        lambdas = [d["lambda"] for d in data]
        syc = [d.get("sycophancy_rate", 0.0) for d in data]
        mc1 = [d.get("mc1", 0.0) for d in data]
        baseline_syc = data[0].get("sycophancy_rate", 0.0)
        baseline_mc1 = data[0].get("mc1", 0.0)
        fig_mitigation_comparison(lambdas, syc, mc1, baseline_syc, baseline_mc1, out_dir)
        print("Wrote fig 4")

    print(f"\nAll figures saved to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
