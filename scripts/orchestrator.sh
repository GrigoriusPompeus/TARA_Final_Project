#!/usr/bin/env bash
# orchestrator.sh - autonomous chain: wait Phase 2 → Phase 3 → figures → stop instance.
# Launch with:
#   ssh ... 'nohup bash /root/tara/scripts/orchestrator.sh \
#       > /root/tara/logs/orchestrator.log 2>&1 < /dev/null & disown'

set -uo pipefail

cd /root/tara
export PATH=/venv/main/bin:$PATH

# Pick up HF_TOKEN + VAST_API_KEY from .env
set -a
. /root/tara/.env
set +a

INSTANCE_ID=39731648
PHASE2_PID=2409

log() { echo "[orch $(date -u +'%Y-%m-%d %H:%M:%S')Z] $*"; }

log "Orchestrator starting. Plan: wait Phase2(PID $PHASE2_PID) -> Phase3 -> figures -> stop instance $INSTANCE_ID."

# === 1. Wait for Phase 2 ===
if kill -0 $PHASE2_PID 2>/dev/null; then
    log "Phase 2 (PID $PHASE2_PID) running. Polling every 60s..."
    while kill -0 $PHASE2_PID 2>/dev/null; do
        sleep 60
    done
    log "Phase 2 process exited."
else
    log "Phase 2 PID $PHASE2_PID not running. Assuming already done."
fi

# === 2. Verify Phase 2 output ===
SUMMARY=/root/tara/results/phase2_tilt/summary.json
if [[ ! -f "$SUMMARY" ]]; then
    log "ERROR: Phase 2 did not produce $SUMMARY. Aborting chain."
    log "Instance LEFT RUNNING so you can debug. Manual stop: vastai stop instance $INSTANCE_ID"
    exit 1
fi
log "Phase 2 produced summary.json. Contents:"
sed 's/^/    /' "$SUMMARY"
log "Files in results/phase2_tilt/:"
ls -la /root/tara/results/phase2_tilt/ | sed 's/^/    /'

# === 3. Phase 3 ===
log "=== Starting Phase 3: BoN sweep (~1-2h expected) ==="
python -u -m scripts.03_bon_sweep \
    --n_prompts all --n_candidates 128 \
    --max_new_tokens 96 --gen_batch 16 --score_batch 8 \
    > /root/tara/logs/phase3.log 2>&1
PHASE3_RC=$?
log "Phase 3 exit code: $PHASE3_RC"
if [[ $PHASE3_RC -ne 0 ]]; then
    log "ERROR: Phase 3 failed. See /root/tara/logs/phase3.log."
    log "Instance LEFT RUNNING for debug."
    exit 1
fi
log "Files in results/phase3_bon/:"
ls -la /root/tara/results/phase3_bon/ 2>/dev/null | sed 's/^/    /' || true

# === 4. Figures ===
log "=== Generating figures ==="
python -u -m scripts.06_make_figures > /root/tara/logs/figures.log 2>&1
FIG_RC=$?
log "Figures exit code: $FIG_RC"
if [[ $FIG_RC -ne 0 ]]; then
    log "WARNING: figures step failed (Phase 2/3 succeeded). Continuing to shutdown anyway."
else
    log "Figures produced:"
    ls -la /root/tara/results/figures/ 2>/dev/null | sed 's/^/    /' || true
fi

# === 5. Stop instance (primary: curl v1; fallback: vastai CLI) ===
log "=== All work done. Stopping Vast instance $INSTANCE_ID ==="

HTTP=$(curl -sS -o /tmp/stop_resp.json -w '%{http_code}' \
    -X PUT \
    -H "Authorization: Bearer $VAST_API_KEY" \
    -H 'Content-Type: application/json' \
    -d '{"state": "stopped"}' \
    "https://console.vast.ai/api/v1/instances/$INSTANCE_ID/" 2>&1 || echo "000")
log "Primary: PUT /api/v1/instances/$INSTANCE_ID/ -> HTTP $HTTP"
[[ -f /tmp/stop_resp.json ]] && sed 's/^/    /' /tmp/stop_resp.json
rm -f /tmp/stop_resp.json

# 200/204 = success; anything else, try fallback
if [[ "$HTTP" != "200" && "$HTTP" != "204" ]]; then
    log "Primary stop failed. Falling back to vastai CLI..."
    /venv/main/bin/vastai stop instance $INSTANCE_ID 2>&1 | sed 's/^/    /'
    log "vastai CLI exit: ${PIPESTATUS[0]}"
fi

log "Orchestrator done at $(date -u +'%Y-%m-%d %H:%M:%S')Z."
log "Results: /root/tara/results/{phase2_tilt,phase3_bon,figures}"
log "To resume: stop -> dashboard -> Start; then ssh in and scp results back to Mac."
