#!/bin/bash
set -e
SCRIPT_DIR=$(dirname ${BASH_SOURCE})
export RESOURCE_NAME="IsaacSim"
export OLD_PYTHONPATH=$PYTHONPATH

# Drop Vulkan loader overrides that point at missing paths; they hide the system NVIDIA ICD
for var in VK_ICD_FILENAMES VK_DRIVER_FILES VK_LAYER_PATH VK_ADD_LAYER; do
    value="${!var}"
    if [ -n "$value" ]; then
        IFS=':' read -ra paths <<< "$value"
        for p in "${paths[@]}"; do
            if [ ! -e "$p" ]; then
                echo "Ignoring $var: '$p' does not exist"
                unset "$var"
                break
            fi
        done
    fi
done

# Duplicate NVIDIA ICD manifests make every GPU enumerate twice, which crashes RTX
if [ -z "$VK_DRIVER_FILES" ] && [ -z "$VK_ICD_FILENAMES" ]; then
    nvidia_icds=()
    for d in /etc/vulkan/icd.d /usr/share/vulkan/icd.d; do
        for f in "$d"/nvidia_icd*.json; do
            [ -e "$f" ] && nvidia_icds+=("$f")
        done
    done
    if [ ${#nvidia_icds[@]} -gt 1 ]; then
        export VK_DRIVER_FILES="${nvidia_icds[0]}"
        echo "Multiple NVIDIA Vulkan ICDs found, using $VK_DRIVER_FILES"
    fi
fi

# Check args for a flag to disable ROS environment setup
NO_ROS_ENV=false
for arg in "$@"; do
    if [ "$arg" == "--no-ros-env" ]; then
        NO_ROS_ENV=true
        echo "Skipping automatic ROS environment setup"
        break
    fi
done

# Data-center GPUs without NVENC need the OpenH264 software encoder (libopenh264 2.4.1) to stream
if [ -z "$STREAMSDK_USE_SOFTWARE_ENCODE" ] && command -v nvidia-smi > /dev/null \
    && nvidia-smi --query-gpu=name --format=csv,noheader -i 0 2>/dev/null | grep -qE "A100|A800|H100|H200|H800"; then
    export STREAMSDK_USE_SOFTWARE_ENCODE=1
    echo "GPU has no NVENC, using software video encoding"
fi

# Source ROS environment setup script if flag was not found
if [ "$NO_ROS_ENV" == "false" ] && [ -f "$SCRIPT_DIR/setup_ros_env.sh" ]; then
    source "$SCRIPT_DIR/setup_ros_env.sh"
fi

exec "$SCRIPT_DIR/kit/kit" "$SCRIPT_DIR/apps/isaacsim.exp.full.streaming.kit" --no-window --/renderer/multiGpu/enabled=false --/renderer/activeGpu=0 "$@"
