"""SycophancyEval loader (Sharma et al. 2023).

Canonical source: github.com/meg-tong/sycophancy-eval (the repo Sharma et al.
released alongside the paper). The HF mirror is incomplete, so we fetch from
the GitHub raw endpoint with on-disk caching.

The 'answer' subset is the QA set used in the proposal V4. Each row contains
an originally neutral factual question; we treat it as a substrate for our
own belief-injection (we don't use Sharma's pre-injected biased prompts so
that the prompt distribution stays controlled).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import requests

from src.config import CFG

GITHUB_RAW_BASE = (
    "https://raw.githubusercontent.com/meg-tong/sycophancy-eval/main/datasets"
)
GITHUB_FILES = {
    "answer": "answer.jsonl",
    "are_you_sure": "are_you_sure.jsonl",
    "feedback": "feedback.jsonl",
    "mimicry": "mimicry.jsonl",
}


def _fetch_jsonl(subset: str) -> Path:
    """Cache the GitHub raw JSONL locally under data/cache/sycophancy_eval/."""
    cache_dir = CFG.paths.data_cache / "sycophancy_eval"
    cache_dir.mkdir(parents=True, exist_ok=True)
    fname = GITHUB_FILES.get(subset)
    if fname is None:
        raise ValueError(f"Unknown SycophancyEval subset {subset!r}")
    cached = cache_dir / fname
    if not cached.exists():
        url = f"{GITHUB_RAW_BASE}/{fname}"
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        cached.write_bytes(r.content)
    return cached


def load_sycophancy_answer(local_path: Path | None = None) -> Iterator[dict]:
    """Yield canonical items from the SycophancyEval 'answer' subset.

    Schema after extraction:
        id, source='sycophancy_eval', question, choices, correct_idx
    """
    if local_path is None:
        try:
            local_path = _fetch_jsonl(CFG.datasets.sycophancy_eval_subset)
        except Exception as e:
            raise RuntimeError(
                "Could not fetch SycophancyEval from GitHub raw; check network. "
                f"Original error: {e}"
            ) from e

    # Each row has prompt (list of turns), base (dict with question /
    # correct_answer / incorrect_answer / answer-list), metadata. We want the
    # CLEAN question from base.question (not the pre-biased prompt) so we
    # can apply our own controlled belief-injection.
    seen_questions: set[str] = set()
    with open(local_path) as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            base = row.get("base") or {}
            q = base.get("question") or row.get("question")
            correct = base.get("correct_answer") or row.get("correct_answer")
            incorrect = base.get("incorrect_answer") or row.get("incorrect_answer")
            if not q or not correct or not incorrect:
                continue
            # SycophancyEval duplicates each base question across multiple
            # bias-injection templates (~5x). We dedupe by question so each
            # canonical item appears once.
            if q in seen_questions:
                continue
            seen_questions.add(q)
            yield {
                "id": f"syc-{i:04d}",
                "source": "sycophancy_eval",
                "question": q,
                "choices": [correct, incorrect],
                "correct_idx": 0,
            }


if __name__ == "__main__":
    items = list(load_sycophancy_answer())
    print(f"Loaded {len(items)} SycophancyEval items")
    if items:
        print("Sample:", items[0])
