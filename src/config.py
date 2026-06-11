"""Central configuration: paths, model IDs, default hyperparameters.

Read with `from src.config import CFG`. Mutate via env vars (TARA_*).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_path(name: str, default: Path) -> Path:
    v = os.environ.get(name)
    return Path(v).expanduser() if v else default


ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Paths:
    root: Path = ROOT
    data_raw: Path = ROOT / "data" / "raw"
    data_processed: Path = ROOT / "data" / "processed"
    data_cache: Path = ROOT / "data" / "cache"
    results: Path = ROOT / "results"
    checkpoints: Path = ROOT / "checkpoints"
    logs: Path = ROOT / "logs"
    configs: Path = ROOT / "configs"
    hf_cache: Path = _env_path("HF_HOME", Path.home() / ".cache" / "huggingface")


@dataclass(frozen=True)
class Models:
    policy: str = "meta-llama/Llama-3.2-1B"
    policy_dtype: str = "bfloat16"
    reward: str = "RLHFlow/ArmoRM-Llama3-8B-v0.1"
    reward_dtype: str = "bfloat16"
    reward_local_quant: str = "8bit"           # for M1 BoN scoring; cloud uses bf16


@dataclass(frozen=True)
class Datasets:
    truthfulqa: str = "truthfulqa/truthful_qa"  # subset 'multiple_choice' for MC1/MC2
    sycophancy_eval_repo: str = "meg-tong/sycophancy-eval"
    sycophancy_eval_subset: str = "answer"      # QA subset, ~1k prompts
    triviaqa: str = "mandarjoshi/trivia_qa"     # supplementary factual substrate


@dataclass(frozen=True)
class BoN:
    n_values: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
    n_candidates_per_prompt: int = 128
    n_agreeing_per_prompt: int = 64             # for tilt measurement (Shapira §6.1)
    n_correcting_per_prompt: int = 64
    max_new_tokens: int = 96
    temperature: float = 1.0
    top_p: float = 1.0
    seed: int = 42


@dataclass(frozen=True)
class GRPO:
    group_size: int = 8
    kl_beta: float = 0.04
    learning_rate: float = 1e-6
    batch_size: int = 16
    n_epochs: int = 2
    max_prompt_length: int = 512
    max_new_tokens: int = 128
    save_every_steps: int = 100
    eval_every_steps: int = 50
    seed: int = 42
    # Shapira Theorem 6 mitigation. Cut to {0, 1.0} per session decision
    # 2026-06-06: full FT throughout, pair the vanilla and corrected runs only.
    lambda_sweep: tuple[float, ...] = (0.0, 1.0)


@dataclass(frozen=True)
class Cfg:
    paths: Paths = field(default_factory=Paths)
    models: Models = field(default_factory=Models)
    datasets: Datasets = field(default_factory=Datasets)
    bon: BoN = field(default_factory=BoN)
    grpo: GRPO = field(default_factory=GRPO)


CFG = Cfg()


def ensure_dirs() -> None:
    """Create gitignored runtime dirs if missing."""
    for p in (
        CFG.paths.data_raw,
        CFG.paths.data_processed,
        CFG.paths.data_cache,
        CFG.paths.results,
        CFG.paths.checkpoints,
        CFG.paths.logs,
    ):
        p.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_dirs()
    print(f"Project root: {CFG.paths.root}")
    print(f"HF cache:     {CFG.paths.hf_cache}")
    print(f"Policy:       {CFG.models.policy}")
    print(f"Reward:       {CFG.models.reward}")
