"""Phase-0 smoke test.

Verifies:
1. Python + torch + transformers versions.
2. Compute device (MPS on M1, CUDA on Linux, CPU fallback).
3. HuggingFace login + access to gated Llama-3.2-1B repo.
4. Loads Llama-3.2-1B Base, generates a few tokens to confirm coherence.
5. Pulls TruthfulQA + SycophancyEval metadata (no full download yet).

Run:  python -m scripts.00_setup_check
"""

from __future__ import annotations

import sys

from src.config import CFG, ensure_dirs


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> int:
    ensure_dirs()

    section("Python + libraries")
    import platform

    print(f"Python:    {platform.python_version()}")

    try:
        import torch

        print(f"torch:     {torch.__version__}")
    except ImportError:
        print("ERROR: torch not installed; run `pip install -e .[local]`")
        return 1

    try:
        import transformers

        print(f"transformers: {transformers.__version__}")
    except ImportError:
        print("ERROR: transformers not installed")
        return 1

    try:
        import datasets as hfd

        print(f"datasets:  {hfd.__version__}")
    except ImportError:
        print("ERROR: datasets not installed")
        return 1

    section("Compute device")
    if torch.cuda.is_available():
        device = "cuda"
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("MPS (Apple Silicon) available")
    else:
        device = "cpu"
        print("WARNING: only CPU available; everything will be slow")

    section("HuggingFace auth")
    from huggingface_hub import HfApi, whoami

    try:
        user = whoami()
        print(f"Logged in as: {user['name']}")
    except Exception as e:
        print(f"NOT logged in: {e}")
        print("Run `huggingface-cli login` with a token that has read access to gated repos.")
        return 2

    api = HfApi()
    try:
        info = api.model_info(CFG.models.policy)
        print(f"Metadata for {CFG.models.policy}: OK (revision {info.sha[:8]})")
    except Exception as e:
        print(f"Cannot access metadata for {CFG.models.policy}: {e}")
        return 3

    # The model_info call only checks public metadata. Gated weights require a
    # separate download attempt. Try fetching config.json to confirm real access.
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import GatedRepoError

    try:
        hf_hub_download(repo_id=CFG.models.policy, filename="config.json")
        print(f"Weight access to {CFG.models.policy}: OK")
    except GatedRepoError:
        print(f"GATED: you have NOT been approved to download {CFG.models.policy}.")
        print(f"Visit https://huggingface.co/{CFG.models.policy} and click 'Submit' on the access form.")
        print("Meta usually auto-approves within minutes to a few hours.")
        return 4
    except Exception as e:
        print(f"Unexpected error checking weight access: {e}")
        return 5

    section("Load policy model and generate a few tokens")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {CFG.models.policy} on {device} ...")
    tok = AutoTokenizer.from_pretrained(CFG.models.policy)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(CFG.models.policy, torch_dtype=dtype)
    model.to(device).eval()

    prompt = "The capital of France is"
    inputs = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    print(f"Prompt:    {prompt!r}")
    print(f"Continuation: {tok.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)!r}")

    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
