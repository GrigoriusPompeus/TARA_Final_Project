"""Build the belief-injection probe dataset (Phase 1).

Reads TruthfulQA-MC (and optionally SycophancyEval) and emits JSONL files
under data/processed/:

    probes_truthfulqa.jsonl       one BiasedPrompt per (source_id, stance, strategy)
    probes_sycophancy.jsonl
    probes_all.jsonl              concatenation, shuffled deterministic

Each row schema (json):
    source_id, source, question, choices (list[str]), correct_idx, injected_idx,
    stance, strategy, text

Note: Are-You-Sure variants are NOT created here because they require a model
turn. They are produced in scripts/02_tilt_measurement.py after first-turn
generation.

Run:    python -m scripts.01_build_dataset
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict

from src.config import CFG, ensure_dirs
from src.data.belief_injection import expand
from src.data.sycophancy_eval import load_sycophancy_answer
from src.data.truthfulqa import load_truthfulqa_mc

SEED = 42


def dump(rows, path) -> int:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(asdict(r) if hasattr(r, "__dataclass_fields__") else r) + "\n")
    return len(rows)


def main() -> int:
    ensure_dirs()

    print("Loading TruthfulQA-MC ...")
    tqa = list(load_truthfulqa_mc(split="validation"))
    print(f"  {len(tqa)} items")

    print("Building TruthfulQA probes ...")
    tqa_probes = expand(tqa, seed=SEED)
    n = dump(tqa_probes, CFG.paths.data_processed / "probes_truthfulqa.jsonl")
    print(f"  wrote {n} prompts to probes_truthfulqa.jsonl")

    # SycophancyEval is best-effort: if the HF mirror is missing we skip with a warning.
    try:
        print("Loading SycophancyEval (answer subset) ...")
        syc = list(load_sycophancy_answer())
        print(f"  {len(syc)} items")
        syc_probes = expand(syc, seed=SEED)
        n = dump(syc_probes, CFG.paths.data_processed / "probes_sycophancy.jsonl")
        print(f"  wrote {n} prompts to probes_sycophancy.jsonl")
    except Exception as e:
        print(f"  SKIP SycophancyEval ({e}); we will rely on belief-injected TruthfulQA only")
        syc_probes = []

    print("Concatenating + shuffling ...")
    all_probes = tqa_probes + syc_probes
    rng = random.Random(SEED)
    rng.shuffle(all_probes)
    n = dump(all_probes, CFG.paths.data_processed / "probes_all.jsonl")
    print(f"  wrote {n} prompts to probes_all.jsonl")

    print("\nBreakdown by source × stance × strategy:")
    counts: dict[tuple[str, str, str], int] = {}
    for p in all_probes:
        key = (p.source, p.stance, p.strategy)
        counts[key] = counts.get(key, 0) + 1
    for k in sorted(counts):
        print(f"  {k}: {counts[k]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
