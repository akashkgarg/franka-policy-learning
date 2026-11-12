#!/bin/bash
# Launch Isaac Sim headless streaming plus the browser-based WebRTC web viewer.
#
# Environment overrides:
#   ISAACSIM_HOST         address the browser uses to reach the stream (default 127.0.0.1)
#   ISAACSIM_SIGNAL_PORT  WebRTC signaling port, TCP (default 49100)
#   ISAACSIM_STREAM_PORT  WebRTC media port, UDP (default 47998)
#   WEB_VIEWER_PORT       web viewer HTTP port (default 8210)
set -e
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE}")" && pwd)
VIEWER_DIR="$SCRIPT_DIR/tools/web-viewer"

export ISAACSIM_HOST="${ISAACSIM_HOST:-127.0.0.1}"
export ISAACSIM_SIGNAL_PORT="${ISAACSIM_SIGNAL_PORT:-49100}"
export ISAACSIM_STREAM_PORT="${ISAACSIM_STREAM_PORT:-47998}"
WEB_VIEWER_PORT="${WEB_VIEWER_PORT:-8210}"

cd "$VIEWER_DIR"
[ -d node_modules ] || npm install --ignore-scripts
# Host and ports are baked into the bundle, so rebuild on every launch
VITE_ISAACSIM_HOST="$ISAACSIM_HOST" \
VITE_ISAACSIM_SIGNAL_PORT="$ISAACSIM_SIGNAL_PORT" \
VITE_ISAACSIM_STREAM_PORT="$ISAACSIM_STREAM_PORT" \
    npm run build --silent
for _ in $(seq 20); do
    curl -s -o /dev/null "http://127.0.0.1:$WEB_VIEWER_PORT/" || break
    sleep 0.5
done
./node_modules/.bin/vite preview --host --port "$WEB_VIEWER_PORT" --strictPort &
VIEWER_PID=$!
trap 'kill $VIEWER_PID 2>/dev/null' EXIT
for _ in $(seq 20); do
    sleep 0.5
    if ! kill -0 $VIEWER_PID 2>/dev/null; then
        echo "Web viewer failed to start on port $WEB_VIEWER_PORT" >&2
        exit 1
    fi
    curl -sf -o /dev/null "http://127.0.0.1:$WEB_VIEWER_PORT/" && break
done
cd "$SCRIPT_DIR"

echo "Web viewer: http://$ISAACSIM_HOST:$WEB_VIEWER_PORT (open once Isaac Sim logs 'app ready')"

"$SCRIPT_DIR/isaac-sim.streaming.sh" \
    --/exts/omni.kit.livestream.app/primaryStream/publicIp="$ISAACSIM_HOST" \
    --/exts/omni.kit.livestream.app/primaryStream/signalPort="$ISAACSIM_SIGNAL_PORT" \
    --/exts/omni.kit.livestream.app/primaryStream/streamPort="$ISAACSIM_STREAM_PORT" \
    "$@"
