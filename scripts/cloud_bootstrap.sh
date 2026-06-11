#!/usr/bin/env bash
# Bootstrap a Vast.ai A100 80GB box for the TARA project.
#
# Workflow (run from your laptop):
#   1. Provision Vast.ai instance with image `pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime`
#      (or any recent CUDA 12.x image with Python 3.10/3.11).
#   2. scp -r src configs scripts pyproject.toml .env user@HOST:~/tara/
#   3. ssh user@HOST, then: bash ~/tara/scripts/cloud_bootstrap.sh
#
# This script:
#   - Verifies python + nvidia-smi.
#   - Creates a venv, installs [cloud,dev] extras.
#   - hf auth login --token "$HF_TOKEN" (from .env).
#   - Runs the setup-check smoke test.

set -euo pipefail

PROJ_DIR="${PROJ_DIR:-$HOME/tara}"
cd "$PROJ_DIR"

# Prefer reusing a venv that already has torch+CUDA wired up (Vast.ai images
# ship /venv/main with the right torch wheel for the box's CUDA). Fall back to
# creating a fresh .venv if no preinstalled venv is present.
PREINSTALLED_VENV="${PREINSTALLED_VENV:-/venv/main}"
if [ -x "$PREINSTALLED_VENV/bin/python" ] && \
   "$PREINSTALLED_VENV/bin/python" -c "import torch" 2>/dev/null; then
    VENV="$PREINSTALLED_VENV"
    echo "[cloud] Reusing preinstalled venv: $VENV"
else
    VENV="$PROJ_DIR/.venv"
    PYBIN="${PYBIN:-python3}"
    if [ ! -d "$VENV" ]; then
        echo "[cloud] Creating venv at $VENV with $PYBIN"
        "$PYBIN" -m venv "$VENV"
    fi
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo "[cloud] Project dir: $PROJ_DIR"
echo "[cloud] Python:      $(python --version) ($(which python))"
echo "[cloud] Hardware:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv | sed 's/^/  /'

echo "[cloud] Upgrading pip"
python -m pip install -U pip wheel

echo "[cloud] Installing cloud extras (transformers, trl, peft, bitsandbytes, wandb, ...)"
pip install -e ".[cloud,dev]"

echo "[cloud] Loading .env"
if [ ! -f .env ]; then
    echo "[cloud] ERROR: .env not found. Copy it from your laptop with scp."
    exit 1
fi
set -a
# shellcheck disable=SC1091
source .env
set +a

echo "[cloud] HF login"
hf auth login --token "$HF_TOKEN" --add-to-git-credential || true

echo "[cloud] Smoke test"
python -m scripts.00_setup_check

echo
echo "[cloud] OK. Suggested next commands:"
echo "  python -m scripts.02_tilt_measurement --n_prompts all --n_per_group 64 --max_new_tokens 96 --gen_batch 16 --score_batch 8 --save_responses"
echo "  python -m scripts.03_bon_sweep --n_prompts all --n_candidates 128 --max_new_tokens 96 --gen_batch 16 --score_batch 8"
echo "  python -m scripts.06_make_figures && ls results/figures/"
