#!/bin/bash
# Wait for runs/<run> to finish training, then evaluate it on the 70 held-out layouts (slots 1-7 x 10 episodes).
#   eval_when_trained.sh <run> <table> [gpus...]
# Results go to ~/franka_llm_data/eval_<run>; bags are not recorded.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE}")" && pwd)
DATA="$HOME/franka_llm_data"
RUN="$1"
TABLE="$2"
shift 2
while pgrep -f "[t]rain_policy.py train .*--out $DATA/runs/$RUN( |$)" > /dev/null; do sleep 60; done
echo "[$(date +%H:%M:%S)] $RUN trained: $(tail -1 "$DATA/runs/$RUN/train_log.jsonl")"
EPISODE_ARGS="--table $TABLE --no-record" "$SCRIPT_DIR/eval_policy.sh" "$DATA/runs/$RUN/policy.pt" \
    "$DATA/eval_$RUN" 10 "$@" > "$DATA/eval_$RUN.log" 2>&1
echo "[$(date +%H:%M:%S)] $RUN: $(tail -1 "$DATA/eval_$RUN.log")"
