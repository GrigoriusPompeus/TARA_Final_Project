#!/usr/bin/env bash
# Sync project to a Vast.ai (or any SSH) host with a non-default port.
# Usage:
#   bash scripts/cloud_sync.sh user@HOST:/path/to/tara PORT
# Example:
#   bash scripts/cloud_sync.sh root@146.115.17.148:/root/tara 41751
#
# Excludes runtime artifacts (.venv, results, checkpoints, papers) so the
# transfer stays small. .env IS copied so the box can authenticate to HF.

set -euo pipefail

DEST="${1:?usage: $0 user@host:/path/to/tara PORT}"
PORT="${2:-22}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "[sync] from $ROOT"
echo "[sync] to   $DEST (port $PORT)"

# -e "ssh -p PORT" tells rsync to tunnel through ssh on the non-standard port.
rsync -avz --progress \
    -e "ssh -p $PORT -o StrictHostKeyChecking=accept-new" \
    --exclude '.venv' \
    --exclude '.claude' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.ipynb_checkpoints' \
    --exclude 'data/cache' \
    --exclude 'data/raw' \
    --exclude 'results' \
    --exclude 'checkpoints' \
    --exclude 'logs' \
    --exclude 'wandb' \
    --exclude 'papers' \
    --exclude 'feedback' \
    --exclude 'proposal' \
    --exclude '*.egg-info' \
    "$ROOT/" \
    "$DEST/"

echo "[sync] done. SSH in with:  ssh -p $PORT ${DEST%:*}"
echo "[sync] Then run:           cd ${DEST##*:} && bash scripts/cloud_bootstrap.sh"
