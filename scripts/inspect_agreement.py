"""Calibrate the agreement detector against actual model responses.

Given `results/phase2_tilt/responses.jsonl` we walk through prompts and show:
    - the user-stated belief
    - the gold answer
    - top-AGREE response with A(x,y) classification
    - top-CORRECT response with A(x,y) classification

This is a sanity check before we trust A(x, y) for the BoN sycophancy curve
or the Theorem 6 corrected reward.

Usage:
    python -m scripts.inspect_agreement                # 10 examples, key_terms
    python -m scripts.inspect_agreement --mode containment --n 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.bon.agreement import agreement_indicator
from src.config import CFG


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", default=str(CFG.paths.results / "phase2_tilt" / "responses.jsonl"))
    ap.add_argument("--mode", default="key_terms", choices=["key_terms", "containment"])
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--probes", default=str(CFG.paths.data_processed / "probes_all.jsonl"))
    args = ap.parse_args()

    # We need the gold answer + user belief per prompt; responses.jsonl only
    # has prompt text. So join with probes_all.jsonl by source_id.
    probe_by_id: dict[str, dict] = {}
    with open(args.probes) as f:
        for line in f:
            r = json.loads(line)
            if r.get("stance") == "belief_wrong" and r.get("strategy") == "answer_suggestion":
                probe_by_id[r["source_id"]] = r

    with open(args.responses) as f:
        rows = [json.loads(line) for line in f]

    rows = rows[: args.n]
    print(f"Inspecting {len(rows)} probes (mode={args.mode}, threshold={args.threshold})\n")

    agree_correct = 0
    agree_total = 0
    correct_correct = 0
    correct_total = 0

    for r in rows:
        sid = r["source_id"]
        probe = probe_by_id.get(sid)
        if probe is None:
            continue
        user_belief = probe["choices"][probe["injected_idx"]]
        correct = probe["choices"][probe["correct_idx"]]

        top_agree = sorted(r["agree"], key=lambda x: -x["score"])[0]
        top_correct = sorted(r["correct"], key=lambda x: -x["score"])[0]

        a_agree = agreement_indicator(
            top_agree["text"], user_belief, correct,
            mode=args.mode, overlap_threshold=args.threshold,
        )
        a_correct = agreement_indicator(
            top_correct["text"], user_belief, correct,
            mode=args.mode, overlap_threshold=args.threshold,
        )

        # We'd EXPECT: top agree-steered to score A=1 ("syco"), top correct-steered to score A=0 ("not syco").
        agree_correct += int(a_agree == 1)
        agree_total += 1
        correct_correct += int(a_correct == 0)
        correct_total += 1

        print(f"--- {sid}  ({r['source']}) ---")
        print(f"  user belief:  {user_belief!r}")
        print(f"  correct:      {correct!r}")
        print(f"  AGREE  top (A={a_agree}): {top_agree['text'][:140]!r}")
        print(f"  CORRECT top (A={a_correct}): {top_correct['text'][:140]!r}")
        print()

    if agree_total:
        print(f"AGREE-steered → A=1 detection rate: {agree_correct}/{agree_total} = {agree_correct/agree_total:.2%}")
    if correct_total:
        print(f"CORRECT-steered → A=0 detection rate: {correct_correct}/{correct_total} = {correct_correct/correct_total:.2%}")
    print("(High AGREE rate + high CORRECT rate means the classifier separates the two distributions cleanly.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
