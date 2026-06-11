"""TruthfulQA loader.

Returns canonical items in the shape expected by `belief_injection.expand`:
    {id, source, question, choices, correct_idx, mc2_targets, mc2_labels}

We use the `multiple_choice` config from the HF dataset because it exposes
both MC1 (single best answer) and MC2 (set of correct answers with labels),
which is what we evaluate truthfulness against — NOT generative exact-match.

Reference:
    Lin, Hilton, Evans (2022). TruthfulQA: Measuring How Models Mimic
    Human Falsehoods. arXiv:2109.07958.
"""

from __future__ import annotations

from typing import Iterator

from datasets import load_dataset


def load_truthfulqa_mc(split: str = "validation") -> Iterator[dict]:
    """Yield canonical items from TruthfulQA's multiple_choice config.

    The HF schema for `truthful_qa/multiple_choice` is:
        question:           str
        mc1_targets.choices:list[str]   <- candidate answers; first is correct
        mc1_targets.labels: list[int]   <- one-hot, the correct index
        mc2_targets.choices:list[str]   <- candidate answers
        mc2_targets.labels: list[int]   <- multi-hot, 1 == correct
    """
    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split=split)
    for i, ex in enumerate(ds):
        mc1_choices = ex["mc1_targets"]["choices"]
        mc1_labels = ex["mc1_targets"]["labels"]
        try:
            mc1_correct = mc1_labels.index(1)
        except ValueError:
            continue  # malformed row, skip

        mc2_choices = ex["mc2_targets"]["choices"]
        mc2_labels = ex["mc2_targets"]["labels"]

        yield {
            "id": f"tqa-{i:04d}",
            "source": "truthfulqa",
            "question": ex["question"],
            "choices": list(mc1_choices),
            "correct_idx": mc1_correct,
            # MC2 evaluation lives alongside MC1 so we can score both metrics
            "mc2_choices": list(mc2_choices),
            "mc2_labels": list(mc2_labels),
        }


if __name__ == "__main__":
    items = list(load_truthfulqa_mc())
    print(f"Loaded {len(items)} TruthfulQA MC items")
    print("Sample:")
    print(items[0])
