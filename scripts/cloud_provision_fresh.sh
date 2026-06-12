#!/usr/bin/env bash
# Bring up a fresh Vast.ai A100 80GB and chain: sync -> bootstrap -> extended Phase 4.
#
# Prereqs you do MANUALLY in the Vast dashboard:
#   1. Destroy the stuck instance (trash icon).
#   2. Search -> filter A100 80GB on-demand -> pick a cheap one (~$1.20-1.50/hr).
#   3. Rent it. Wait for Status: Running.
#   4. From the instance row, copy the SSH command (Direct SSH).
#   5. Update the INSTANCE_ID below with the new instance ID from the dashboard.
#
# Then run from your Mac:
#   bash scripts/cloud_provision_fresh.sh root@HOST PORT INSTANCE_ID
#
# Example:
#   bash scripts/cloud_provision_fresh.sh root@123.45.67.89 12345 39812345
#
# This script:
#   1. Verifies SSH works.
#   2. rsyncs repo + .env to /root/tara on the instance.
#   3. Runs cloud_bootstrap.sh (creates venv, installs deps, HF login, sanity check).
#   4. Patches orchestrator_extended.sh with the new INSTANCE_ID.
#   5. Launches the orchestrator under nohup.
#   6. Tails the log for 30 s so you see it starting; then detaches.
#
# After it returns, the cloud job runs to completion on its own and auto-stops
# the instance on success. You poll/scp from your Mac when ready.

set -euo pipefail

DEST="${1:?usage: $0 root@HOST PORT INSTANCE_ID}"
PORT="${2:?usage: $0 root@HOST PORT INSTANCE_ID}"
INSTANCE_ID="${3:?usage: $0 root@HOST PORT INSTANCE_ID}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOST="${DEST#*@}"
USER="${DEST%@*}"

echo "[provision] dest=$DEST port=$PORT instance=$INSTANCE_ID"

# === 1. SSH probe ===
echo "[provision] 1/5 SSH probe..."
ssh -p "$PORT" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 "$DEST" \
    "hostname && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader" \
    || { echo "[provision] SSH failed."; exit 1; }

# === 2. Rsync ===
echo "[provision] 2/5 rsync repo + .env to $DEST:/root/tara"
bash "$ROOT/scripts/cloud_sync.sh" "$DEST:/root/tara" "$PORT"

# === 3. Bootstrap ===
echo "[provision] 3/5 cloud_bootstrap.sh"
ssh -p "$PORT" "$DEST" \
    "cd /root/tara && bash scripts/cloud_bootstrap.sh"

# === 4. Patch orchestrator INSTANCE_ID and launch ===
echo "[provision] 4/5 patch orchestrator_extended.sh with INSTANCE_ID=$INSTANCE_ID"
ssh -p "$PORT" "$DEST" \
    "sed -i 's/^INSTANCE_ID=.*/INSTANCE_ID=$INSTANCE_ID/' /root/tara/scripts/orchestrator_extended.sh && grep '^INSTANCE_ID=' /root/tara/scripts/orchestrator_extended.sh"

echo "[provision] 5/5 launching orchestrator_extended.sh under nohup"
ssh -p "$PORT" "$DEST" \
    "mkdir -p /root/tara/logs && \
     nohup bash /root/tara/scripts/orchestrator_extended.sh \
        > /root/tara/logs/orchestrator_extended.log 2>&1 < /dev/null & \
     disown && \
     sleep 2 && \
     echo '[provision] orchestrator launched, PID:' \$(pgrep -f orchestrator_extended.sh)"

echo "[provision] tailing orchestrator log for 30s so you see it starting..."
ssh -p "$PORT" "$DEST" "timeout 30 tail -f /root/tara/logs/orchestrator_extended.log" || true

echo
echo "[provision] DONE. Training runs in background on cloud."
echo
echo "To check progress:"
echo "  ssh -p $PORT $DEST 'tail -f /root/tara/logs/grpo_vanilla_extended.log'"
echo
echo "To scp the trained model back when it's done:"
echo "  scp -P $PORT -r $DEST:/root/tara/checkpoints/grpo_vanilla_extended/final \\"
echo "    \"$ROOT/checkpoints/\""
echo
echo "On success the orchestrator auto-stops instance $INSTANCE_ID."
echo "On failure it leaves the instance running for debug; check dashboard."
