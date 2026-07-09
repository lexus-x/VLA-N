#!/bin/bash
# ponytail: polls hosts serially-parallel every $INTERVAL s over existing SSH config.
# swap for dcgm-exporter+Grafana only if this needs to scale past a handful of boxes.
HOSTS=(a100 blackwell a6000 blackwell2 a6000-left a6000-mid)
INTERVAL="${1:-5}"

while true; do
  tmpdir=$(mktemp -d)
  for h in "${HOSTS[@]}"; do
    ssh -o ConnectTimeout=5 -o BatchMode=yes "$h" \
      "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits" \
      > "$tmpdir/$h" 2>/dev/null &
  done
  wait

  clear
  echo "GPU Dashboard  $(date '+%Y-%m-%d %H:%M:%S')  (refresh ${INTERVAL}s, Ctrl+C to quit)"
  echo
  printf "%-12s %-4s %-32s %16s %6s\n" "HOST" "GPU" "NAME" "MEM USED/TOTAL" "UTIL"
  for h in "${HOSTS[@]}"; do
    if [ -s "$tmpdir/$h" ]; then
      while IFS=',' read -r idx name used total util; do
        idx=$(echo "$idx" | xargs); name=$(echo "$name" | xargs | cut -c1-32)
        used=$(echo "$used" | xargs); total=$(echo "$total" | xargs); util=$(echo "$util" | xargs)
        printf "%-12s %-4s %-32s %8s/%-7s %5s%%\n" "$h" "$idx" "$name" "${used}MiB" "${total}MiB" "$util"
      done < "$tmpdir/$h"
    else
      printf "%-12s %-4s %-32s %16s %6s\n" "$h" "-" "UNREACHABLE" "-" "-"
    fi
  done
  rm -rf "$tmpdir"

  sleep "$INTERVAL"
done
