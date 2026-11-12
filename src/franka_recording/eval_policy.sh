#!/bin/bash
# Closed-loop evaluation of a trained policy on held-out seeds (900000+), one sim + policy server per slot.
#   eval_policy.sh <checkpoint> <out_dir> [episodes_per_slot] [gpus...]
# Slot s (1..7) always evaluates seeds 900000 + s*1000 + i, so every eval sees the same layouts; slots are spread
# over the given GPUs round-robin (default: slot s on GPU s).
# EPISODE_ARGS is passed through to franka_llm_episodes.py, e.g. EPISODE_ARGS="--table wood".
# SLOTS overrides the slot list (default "1 2 3 4 5 6 7").
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE}")" && pwd)
ISAAC_SIM_DIR=${ISAAC_SIM_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}
PYTHON="${POLICY_PYTHON:-$HOME/.venvs/franka-policy/bin/python}"
CHECKPOINT="$1"
OUT="$2"
EPISODES="${3:-8}"
shift 3 2>/dev/null
GPUS=("$@")
[ ${#GPUS[@]} -eq 0 ] && GPUS=(1 2 3 4 5 6 7)
read -r -a SLOT_LIST <<< "${SLOTS:-1 2 3 4 5 6 7}"
mkdir -p "$OUT/logs"
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export VK_DRIVER_FILES=${VK_DRIVER_FILES:-/etc/vulkan/icd.d/nvidia_icd.json}

gpu_for() { echo "${GPUS[$(( ($1 - 1) % ${#GPUS[@]} ))]}"; }

SERVERS=()
SIMS=()
for SLOT in "${SLOT_LIST[@]}"; do
    GPU=$(gpu_for "$SLOT")
    CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" "$SCRIPT_DIR/train_policy.py" serve --checkpoint "$CHECKPOINT" \
        --address 127.0.0.1:$((47720 + SLOT)) > "$OUT/logs/server_slot$SLOT.log" 2>&1 &
    SERVERS+=($!)
done
trap 'kill ${SERVERS[@]} 2>/dev/null' EXIT
sleep 20
for SLOT in "${SLOT_LIST[@]}"; do
    GPU=$(gpu_for "$SLOT")
    ROS_DOMAIN_ID=$((80 + SLOT)) "$ISAAC_SIM_DIR/python.sh" "$SCRIPT_DIR/franka_llm_episodes.py" --mode policy \
        --policy-address 127.0.0.1:$((47720 + SLOT)) --out "$OUT" --episodes "$EPISODES" \
        --seed $((900000 + SLOT * 1000)) --gpu "$GPU" $EPISODE_ARGS > "$OUT/logs/sim_slot$SLOT.log" 2>&1 &
    SIMS+=($!)
    sleep 2
done
wait "${SIMS[@]}"
python3 - "$OUT" <<'EOF'
import glob, json, sys
episodes = [json.load(open(f)) for f in sorted(glob.glob(f"{sys.argv[1]}/*/episode.json"))]
wins = sum(e["success"] for e in episodes)
print(f"policy success: {wins}/{len(episodes)} = {wins / max(1, len(episodes)):.1%}")
json.dump({"episodes": len(episodes), "successes": wins,
           "results": [{k: e[k] for k in ("episode", "task", "success", "xy_error_m", "height_diff_m")} for e in episodes]},
          open(f"{sys.argv[1]}/summary.json", "w"), indent=1)
EOF
