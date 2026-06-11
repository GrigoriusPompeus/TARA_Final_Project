"""Training dataset for GRPO.

Each row carries:
    - `prompt`: the input fed to the policy (few-shot scaffolded biased question)
    - `correct_answer`: gold string used for mitigation reward correction
    - `user_belief`: the stated belief used for A(x, y) classification
    - `is_false`: 1 if x' ∈ X_false (belief_wrong), 0 otherwise
    - `source_id`, `source`: for stratified eval

The Shapira Theorem 6 corrected reward is computed at training time inside
`src.grpo.reward_fn` using `is_false` × A(x, y).

For the GRPO training pool we use a mix:
    - belief_wrong probes  (X_false → optimization pressure target)
    - belief_correct probes (X_true  → control: should NOT amplify on these)
    - neutral probes        (no belief injected; teaches the policy to remain
                            coherent under regular questions and prevents
                            entropy collapse to a uniform "yes" response)

The dataset is held in memory; this works for our ~3-6k probe scale.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from datasets import Dataset

from src.config import CFG
from src.data.few_shot import wrap_query


def _row_to_record(row: dict) -> dict:
    correct = row["choices"][row["correct_idx"]]
    user_belief = (
        row["choices"][row["injected_idx"]]
        if row["injected_idx"] is not None
        else ""
    )
    is_false = 1 if row["stance"] == "belief_wrong" else 0
    return {
        "prompt": wrap_query(row["text"]),
        "correct_answer": correct,
        "user_belief": user_belief,
        "is_false": is_false,
        "stance": row["stance"],
        "strategy": row["strategy"],
        "source_id": row["source_id"],
        "source": row["source"],
    }


def load_probes(
    probes_path: Path | None = None,
    *,
    strata: tuple[str, ...] = ("neutral", "belief_correct", "belief_wrong"),
    n_per_stratum: int | None = None,
    seed: int = 42,
) -> list[dict]:
    """Load probes and balance across strata."""
    probes_path = probes_path or CFG.paths.data_processed / "probes_all.jsonl"
    rng = random.Random(seed)
    by_stance: dict[str, list[dict]] = {s: [] for s in strata}
    with open(probes_path) as f:
        for line in f:
            row = json.loads(line)
            if row["stance"] in by_stance:
                by_stance[row["stance"]].append(row)
    for s in by_stance:
        rng.shuffle(by_stance[s])
        if n_per_stratum is not None:
            by_stance[s] = by_stance[s][:n_per_stratum]
    combined: list[dict] = []
    for s in strata:
        combined.extend(by_stance[s])
    rng.shuffle(combined)
    return [_row_to_record(r) for r in combined]


def to_hf_dataset(rows: list[dict]) -> Dataset:
    return Dataset.from_list(rows)


def train_eval_split(
    rows: list[dict],
    eval_frac: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    n_eval = max(1, int(len(shuffled) * eval_frac))
    return shuffled[n_eval:], shuffled[:n_eval]
