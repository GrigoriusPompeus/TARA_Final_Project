#!/usr/bin/env bash
# Extended Phase 4: longer vanilla GRPO to push policy past the noise floor.
#
# Knobs vs original Phase 4:
#   learning_rate  1e-6  -> 3e-6   (3x)
#   n_epochs       2     -> 4      (2x)
#   save_every     100   -> 1000   (less disk pressure)
#   mitigation_lambda    0          (vanilla)
#   kl_beta              0.04       (unchanged)
#
# Expected KL drift ~ 0.03-0.05 (6x policy movement vs original 0.005 KL).
# Wall-clock: ~90 min on A100 80GB.
#
# Auto-stop policy:
#   - training success  -> stop instance (via curl PUT -> vastai CLI fallback)
#   - training failure  -> LEAVE instance running for debug, exit non-zero
#
# Launch with:
#   ssh -p <PORT> root@<VAST_HOST> \
#     'nohup bash /root/tara/scripts/orchestrator_extended.sh \
#        > /root/tara/logs/orchestrator_extended.log 2>&1 < /dev/null & disown'

set -uo pipefail

cd /root/tara
export PATH=/venv/main/bin:$PATH

set -a
. /root/tara/.env
set +a

INSTANCE_ID="${INSTANCE_ID:-<vast-instance-id>}"
RUN_NAME=grpo_vanilla_extended
TRAIN_LOG=/root/tara/logs/${RUN_NAME}.log
OUT_DIR=/root/tara/checkpoints/${RUN_NAME}

log() { echo "[orch $(date -u +'%Y-%m-%d %H:%M:%S')Z] $*"; }

log "Extended Phase 4 orchestrator starting on instance $INSTANCE_ID."
log "Disk before training:"
df -h /root | sed 's/^/    /'

# === 1. Run extended Phase 4 ===
log "=== Starting Phase 4 extended: ${RUN_NAME} ==="
log "Args: --learning_rate 3e-6 --n_epochs 4 --save_every 1000 --run_name $RUN_NAME"

python -u -m scripts.04_grpo_train \
    --run_name "$RUN_NAME" \
    --learning_rate 3e-6 \
    --n_epochs 4 \
    --save_every 1000 \
    --mitigation_lambda 0.0 \
    > "$TRAIN_LOG" 2>&1
RC=$?
log "Training exit code: $RC"

if [[ $RC -ne 0 ]]; then
    log "ERROR: training failed. See $TRAIN_LOG."
    log "Instance LEFT RUNNING for debug. Manual stop: vastai stop instance $INSTANCE_ID"
    log "Last 30 lines of training log:"
    tail -30 "$TRAIN_LOG" | sed 's/^/    /'
    exit 1
fi

# === 2. Verify final checkpoint ===
if [[ ! -f "$OUT_DIR/final/model.safetensors" ]]; then
    log "ERROR: training reported success but $OUT_DIR/final/model.safetensors missing."
    log "Contents of $OUT_DIR:"
    ls -la "$OUT_DIR" 2>&1 | sed 's/^/    /'
    log "Instance LEFT RUNNING for debug."
    exit 1
fi

log "Final checkpoint OK: $OUT_DIR/final/"
ls -lh "$OUT_DIR/final/" | sed 's/^/    /'

# === 3. Stop instance (primary: curl v1 API; fallback: vastai CLI) ===
log "=== Training done. Stopping Vast instance $INSTANCE_ID ==="

HTTP=$(curl -sS -o /tmp/stop_resp.json -w '%{http_code}' \
    -X PUT \
    -H "Authorization: Bearer $VAST_API_KEY" \
    -H 'Content-Type: application/json' \
    -d '{"state": "stopped"}' \
    "https://console.vast.ai/api/v1/instances/$INSTANCE_ID/" 2>&1 || echo "000")
log "Primary stop: PUT /api/v1/instances/$INSTANCE_ID/ -> HTTP $HTTP"
[[ -f /tmp/stop_resp.json ]] && sed 's/^/    /' /tmp/stop_resp.json
rm -f /tmp/stop_resp.json

if [[ "$HTTP" != "200" && "$HTTP" != "204" ]]; then
    log "Primary stop failed. Falling back to vastai CLI..."
    /venv/main/bin/vastai stop instance $INSTANCE_ID 2>&1 | sed 's/^/    /'
    log "vastai CLI exit: ${PIPESTATUS[0]}"
fi

log "Orchestrator done at $(date -u +'%Y-%m-%d %H:%M:%S')Z."
log "Output: $OUT_DIR/final/  (scp back to Mac for eval)"
