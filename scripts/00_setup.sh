#!/usr/bin/env bash
# One-shot environment setup for the TARA project.
# Run from the project root:
#   bash scripts/00_setup.sh

set -euo pipefail

PYBIN="${PYBIN:-python3.11}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "[setup] Project root: $ROOT"
echo "[setup] Python: $($PYBIN --version)"

if [ ! -d .venv ]; then
    echo "[setup] Creating venv at .venv"
    "$PYBIN" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo "[setup] Upgrading pip"
python -m pip install -U pip wheel

# Detect platform
UNAME="$(uname -s)"
ARCH="$(uname -m)"
if [ "$UNAME" = "Darwin" ] && [ "$ARCH" = "arm64" ]; then
    EXTRAS="local,dev"
    echo "[setup] Detected Apple Silicon — installing [local,dev]"
else
    EXTRAS="cloud,dev"
    echo "[setup] Detected non-Mac — installing [cloud,dev]"
fi

echo "[setup] Installing project (this can take a few minutes the first time)"
pip install -e ".[$EXTRAS]"

echo "[setup] Done. Activate with:  source .venv/bin/activate"
echo "[setup] Next:  python -m scripts.00_setup_check"
