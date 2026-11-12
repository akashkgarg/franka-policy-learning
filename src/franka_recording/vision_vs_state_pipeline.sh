#!/bin/bash
# Train a policy on the vision-only (wood table) LLM dataset and compare it against the policies trained on the
# full-state LLM dataset, on identical held-out layouts (seeds 900000+).
#   vision_vs_state_pipeline.sh [episodes_per_gpu] [gpus...]
# Waits for the wood collection workers to exit, converts the bags, trains runs/wood_vision (images + joints + task,
# goal actions, same hyperparameters as runs/goal_vision), then evaluates every policy on both tables.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE}")" && pwd)
DATA="$HOME/franka_llm_data"
PYTHON="$HOME/.venvs/franka-policy/bin/python"
EPISODES="${1:-10}"
shift 1 2>/dev/null
GPUS=("$@")
[ ${#GPUS[@]} -eq 0 ] && GPUS=(1 2 3 4 5 6 7)
log() { echo "[$(date +%H:%M:%S)] $*"; }

log "waiting for the wood collection to finish"
while pgrep -f "[c]ollect_llm_worker" > /dev/null; do sleep 60; done

log "converting $DATA/vision/wood_episodes"
bash -c "source /opt/ros/jazzy/setup.bash && python3.12 '$SCRIPT_DIR/bags_to_dataset.py' \
    '$DATA/vision/wood_episodes' '$DATA/dataset_vision_wood' --workers 16" > "$DATA/dataset_vision_wood.log" 2>&1
log "dataset: $(ls "$DATA/dataset_vision_wood" | grep -c npz) episodes"

if [ ! -f "$DATA/runs/wood_vision/policy.pt" ]; then
    log "training runs/wood_vision"
    CUDA_VISIBLE_DEVICES=${GPUS[0]} "$PYTHON" "$SCRIPT_DIR/train_policy.py" train --data "$DATA/dataset_vision_wood" \
        --out "$DATA/runs/wood_vision" --action goal --steps 20000 > "$DATA/runs/wood_vision.log" 2>&1 || exit 1
fi
log "trained: $(tail -1 "$DATA/runs/wood_vision/train_log.jsonl")"

# name checkpoint table record [episodes per gpu]; bags are ~0.9 GB per episode, so only a small replay set is recorded
EVALS=(
    "eval_wood_vision wood_vision wood no"
    "eval_wood_vision_replays wood_vision wood yes 2"
    "eval_cmp_goal_state_grid goal_state grid no"
    "eval_cmp_goal_vision_grid goal_vision grid no"
    "eval_cmp_goal_state_wood goal_state wood no"
    "eval_cmp_goal_vision_wood goal_vision wood no"
    "eval_cmp_wood_vision_grid wood_vision grid no"
)
for spec in "${EVALS[@]}"; do
    read -r NAME CHECKPOINT TABLE RECORD COUNT <<< "$spec"
    [ -f "$DATA/$NAME/summary.json" ] && continue
    ARGS="--table $TABLE"
    [ "$RECORD" = no ] && ARGS="$ARGS --no-record"
    log "evaluating $CHECKPOINT on the $TABLE table -> $NAME"
    EPISODE_ARGS="$ARGS" "$SCRIPT_DIR/eval_policy.sh" "$DATA/runs/$CHECKPOINT/policy.pt" "$DATA/$NAME" "${COUNT:-$EPISODES}" \
        "${GPUS[@]}" > "$DATA/$NAME.log" 2>&1
    log "$(tail -1 "$DATA/$NAME.log")"
done

python3 - "$DATA" <<'EOF'
import json, sys
data = sys.argv[1]
rows = [("wood_vision", "vision-only LLM data (wood)", "images + joints + task"),
        ("goal_state", "full-state LLM data (grid)", "images + joints + task + block positions"),
        ("goal_vision", "full-state LLM data (grid)", "images + joints + task")]
names = {("wood_vision", "wood"): "eval_wood_vision", ("wood_vision", "grid"): "eval_cmp_wood_vision_grid",
         ("goal_state", "grid"): "eval_cmp_goal_state_grid", ("goal_state", "wood"): "eval_cmp_goal_state_wood",
         ("goal_vision", "grid"): "eval_cmp_goal_vision_grid", ("goal_vision", "wood"): "eval_cmp_goal_vision_wood"}
table = []
for policy, trained_on, inputs in rows:
    row = {"policy": policy, "trained_on": trained_on, "inputs": inputs}
    for surface in ("wood", "grid"):
        try:
            s = json.load(open(f"{data}/{names[policy, surface]}/summary.json"))
            row[surface] = f"{s['successes']}/{s['episodes']}"
        except FileNotFoundError:
            row[surface] = None
    table.append(row)
json.dump(table, open(f"{data}/vision_vs_state_comparison.json", "w"), indent=1)
for row in table:
    print(f"{row['policy']:12s} {row['trained_on']:30s} {row['inputs']:42s} wood {row['wood']}  grid {row['grid']}")
EOF
log "done"
