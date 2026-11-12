#!/bin/bash
# Run the Franka pick-and-place sim and record every topic into a ROS 2 bag with MCAP storage.
#
# Usage: record_franka_bag.sh [output_dir] [-- extra args for franka_pick_place_ros2.py]
# Output: <output_dir>/metadata.yaml and <output_dir>/<name>_0.mcap
set -e
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE}")" && pwd)
ISAAC_SIM_DIR=${ISAAC_SIM_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}

OUT="${1:-$HOME/franka_recordings/franka_pick_place_$(date +%Y%m%d_%H%M%S)}"
shift || true
[ "$1" == "--" ] && shift
mkdir -p "$(dirname "$OUT")"

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

# Isaac Sim uses its bundled ROS 2 libraries; only the recorder sources the system install.
# /usr/bin/python3 is not 3.12 on this image, so call the Jazzy CLI with python3.12 directly.
bash -c 'source /opt/ros/jazzy/setup.bash
         exec python3.12 /opt/ros/jazzy/bin/ros2 bag record --all --storage mcap \
             --storage-preset-profile zstd_fast --max-cache-size 1073741824 --use-sim-time \
             --output "$0"' "$OUT" &
RECORDER_PID=$!
# Background jobs of a non-interactive shell ignore SIGINT, so stop the recorder with SIGTERM
trap 'kill -TERM $RECORDER_PID 2>/dev/null; wait $RECORDER_PID 2>/dev/null' EXIT
sleep 3

SIM_STATUS=0
VK_DRIVER_FILES="${VK_DRIVER_FILES:-/etc/vulkan/icd.d/nvidia_icd.json}" \
    "$ISAAC_SIM_DIR/python.sh" "$SCRIPT_DIR/franka_pick_place_ros2.py" "$@" || SIM_STATUS=$?

sleep 2
kill -TERM $RECORDER_PID
wait $RECORDER_PID || true
trap - EXIT

bash -c 'source /opt/ros/jazzy/setup.bash; python3.12 /opt/ros/jazzy/bin/ros2 bag info "$0"' "$OUT"
echo "Bag written to $OUT (sim exit status $SIM_STATUS)"
exit $SIM_STATUS
