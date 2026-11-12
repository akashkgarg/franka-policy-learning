#!/bin/bash
# Serve the Franka LLM Lab. Usage: run.sh [--port 8240] [--gpu 1] [--data ~/franka_llm_data]
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE}")" && pwd)
PORT=8240
for ((i = 1; i <= $#; i++)); do [ "${!i}" == "--port" ] && j=$((i + 1)) && PORT="${!j}"; done
if [ -n "$CODER_WORKSPACE_NAME" ] && [ -n "$CODER_WORKSPACE_OWNER_NAME" ]; then
    echo "Open https://$PORT--standard--$CODER_WORKSPACE_NAME--$CODER_WORKSPACE_OWNER_NAME.devspaces.rbx.com"
fi
echo "Open http://localhost:$PORT (or forward port $PORT)"
source /opt/ros/jazzy/setup.bash
exec python3.12 "$SCRIPT_DIR/server.py" "$@"
