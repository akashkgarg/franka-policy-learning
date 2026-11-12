#!/bin/bash
# Collect LLM-driven Franka stacking episodes in parallel, one Isaac Sim worker per GPU.
#   collect_llm_episodes.sh <out_dir> [episodes_per_worker] [gpus...]
# Each worker uses its own ROS domain, so bag recorders only see their own simulation. Workers resume
# where they left off (episodes with an episode.json are skipped), and are restarted if Isaac Sim exits.
# EPISODE_ARGS is passed through to franka_llm_episodes.py, e.g. EPISODE_ARGS="--observation vision".
# Stop with: pkill -f collect_llm_worker
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE}")" && pwd)
ISAAC_SIM_DIR=${ISAAC_SIM_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}
OUT="${1:-$HOME/franka_llm_data/episodes}"
EPISODES="${2:-1000}"
shift 2 2>/dev/null
GPUS=("$@")
[ ${#GPUS[@]} -eq 0 ] && GPUS=(1 2 3 4 5 6 7)
mkdir -p "$OUT/logs"

for GPU in "${GPUS[@]}"; do
    setsid nohup bash -c '
        exec -a collect_llm_worker bash -c "
            export ROS_DOMAIN_ID=\$((60 + $0)) ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST RMW_IMPLEMENTATION=rmw_fastrtps_cpp
            export VK_DRIVER_FILES=\${VK_DRIVER_FILES:-/etc/vulkan/icd.d/nvidia_icd.json}
            for attempt in 1 2 3 4 5 6 7 8 9 10; do
                \"$1/python.sh\" \"$2/franka_llm_episodes.py\" --out \"$3\" --episodes $4 --seed \$(($0 * 100000)) --gpu $0 $5 && break
                echo \"[worker] Isaac Sim exited with \$?, restarting (attempt \$attempt)\"
                sleep 5
            done"
    ' "$GPU" "$ISAAC_SIM_DIR" "$SCRIPT_DIR" "$OUT" "$EPISODES" "$EPISODE_ARGS" > "$OUT/logs/gpu$GPU.log" 2>&1 &
    echo "worker on GPU $GPU -> $OUT/logs/gpu$GPU.log"
    sleep 2
done
