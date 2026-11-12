#!/bin/bash
# Evaluate the image-resolution sweep (runs/wood_<variant>) on the wood table as each training run finishes.
# Every eval covers the same 70 held-out layouts as eval_wood_vision (slots 1-7 x 10 episodes).
#   resolution_sweep_eval.sh
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE}")" && pwd)
DATA="$HOME/franka_llm_data"
log() { echo "[$(date +%H:%M:%S)] $*"; }

evaluate() {  # <variant> <gpus...>
    local VARIANT=$1
    shift
    local NAME="eval_wood_$VARIANT"
    [ -f "$DATA/$NAME/summary.json" ] && return
    while pgrep -f "[t]rain_policy.py train --data $DATA/dataset_wood_$VARIANT " > /dev/null; do sleep 60; done
    log "$VARIANT trained: $(tail -1 "$DATA/runs/wood_$VARIANT/train_log.jsonl")"
    log "evaluating $VARIANT on GPUs $*"
    EPISODE_ARGS="--table wood --no-record" "$SCRIPT_DIR/eval_policy.sh" "$DATA/runs/wood_$VARIANT/policy.pt" \
        "$DATA/$NAME" 10 "$@" > "$DATA/$NAME.log" 2>&1
    log "$VARIANT: $(tail -1 "$DATA/$NAME.log")"
}

# GPU 5 trains the full-resolution model for hours and GPU 4 the half-resolution one, so keep sims off them
evaluate res256 1 2 3 6 7
evaluate crop264 1 2 3 6 7
evaluate res320 1 2 3 4 6 7
evaluate res640 1 2 3 4 5 6 7

python3 - "$DATA" <<'EOF'
import json, sys
data = sys.argv[1]
rows = [("wood_vision", "128x128 (squashed)", "128x128", "eval_wood_vision"),
        ("wood_res256", "256x192", "256x192", "eval_wood_res256"),
        ("wood_crop264", "table crop -> 264x176", "256x192", "eval_wood_crop264"),
        ("wood_res320", "320x240", "320x240", "eval_wood_res320"),
        ("wood_res640", "640x480 (full)", "640x480", "eval_wood_res640")]
table = []
for run, scene, wrist, name in rows:
    try:
        s = json.load(open(f"{data}/{name}/summary.json"))
        result = f"{s['successes']}/{s['episodes']}"
    except FileNotFoundError:
        result = None
    log = [json.loads(l) for l in open(f"{data}/runs/{run}/train_log.jsonl") if "val_l1" in l]
    table.append({"run": run, "scene": scene, "wrist": wrist, "wood_success": result,
                  "best_val_l1": min(l["val_l1"] for l in log), "train_seconds": log[-1]["elapsed_s"]})
    print(f"{run:14s} scene {scene:24s} wrist {wrist:9s} success {result}  val_l1 {table[-1]['best_val_l1']}")
json.dump(table, open(f"{data}/resolution_sweep.json", "w"), indent=1)
EOF
log "done"
