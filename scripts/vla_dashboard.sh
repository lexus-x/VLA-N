#!/bin/bash
# ponytail: single-file poll-loop dashboard; swap for a real TUI/Grafana only if this needs more than eyeballing.
HOST="a6000-left"
RESULT_FILE='C:\Users\islab01\vla-atlas\experiments\firstpass_sweep\decoder_fix_validation_v2.json'
INTERVAL="${1:-5}"

while true; do
  clear
  echo "VLA Dashboard  $(date '+%Y-%m-%d %H:%M:%S')  (refresh ${INTERVAL}s, Ctrl+C to quit)"
  echo

  gpu=$(ssh -o ConnectTimeout=5 -o BatchMode=yes "$HOST" \
    "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits" 2>/dev/null)
  echo "GPU (a6000-left):"
  echo "$gpu" | awk -F',' '{printf "  GPU%s  %5s/%-6s MiB  %3s%% util\n", $1, $2, $3, $4}'
  echo

  proc=$(ssh -o ConnectTimeout=5 -o BatchMode=yes "$HOST" "tasklist | findstr /i python" 2>/dev/null)
  n_proc=$(echo "$proc" | grep -c python)
  echo "Python processes on a6000-left: $n_proc"
  echo

  results=$(ssh -o ConnectTimeout=5 -o BatchMode=yes "$HOST" "type $RESULT_FILE" 2>/dev/null)
  n_done=$(echo "$results" | grep -c '"cell_id"')
  echo "decoder_fix_validation_v2.json: $n_done/4 cells done"
  echo "$results" | grep -E '"cell_id"|"success_rate"|"mean_jerk"' | paste -d' ' - - - | \
    sed -E 's/.*"cell_id": "([^"]+)".*"success_rate": ([0-9.]+).*"mean_jerk": ([0-9.]+).*/  \1  success=\2  jerk=\3/'

  sleep "$INTERVAL"
done
