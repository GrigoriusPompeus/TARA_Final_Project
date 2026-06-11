"""Run TruthfulQA MC1/MC2 eval on a (possibly fine-tuned) policy.

For each TQA-MC validation item we compute, under the policy:
    log P(choice | question)   for every mc1 choice and every mc2 choice
and aggregate via `src.eval.mc.mc1_mc2`. Returns averaged MC1 and MC2.

This is the "gold" truthfulness signal that lives next to the ArmoRM "proxy"
reward across all phases.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.data.truthfulqa import load_truthfulqa_mc
from src.eval.mc import mc1_mc2
from src.models.policy import Policy, batched_logprob_of_completions


@dataclass
class MCRun:
    mc1_mean: float
    mc2_mean: float
    n: int


def run(
    policy: Policy,
    *,
    split: str = "validation",
    n_items: int | None = None,
    use_few_shot_prefix: bool = True,
) -> MCRun:
    from src.data.few_shot import build_prefix

    prefix = build_prefix() if use_few_shot_prefix else ""
    items = list(load_truthfulqa_mc(split=split))
    if n_items is not None:
        items = items[:n_items]
    mc1_sum = 0.0
    mc2_sum = 0.0
    n = 0
    for it in items:
        prompt = f"{prefix}Q: {it['question']}\nA: "
        lp_mc1 = batched_logprob_of_completions(policy, prompt, it["choices"])
        lp_mc2 = batched_logprob_of_completions(policy, prompt, it["mc2_choices"])
        r = mc1_mc2(lp_mc1, it["correct_idx"], lp_mc2, it["mc2_labels"])
        mc1_sum += r.mc1
        mc2_sum += r.mc2
        n += 1
    return MCRun(mc1_mean=mc1_sum / n, mc2_mean=mc2_sum / n, n=n)
