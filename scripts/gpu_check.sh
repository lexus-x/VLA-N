#!/bin/bash
# One-shot GPU snapshot across all 6 lab hosts (exits immediately, no loop).
# For live refresh use gpu_dashboard.sh instead.
# FREE = <2GB used AND <10% util. a6000-left GPU0 is reserved (fishonet) -> flagged.
HOSTS=(a100 blackwell a6000 blackwell2 a6000-left a6000-mid)

tmpdir=$(mktemp -d)
for h in "${HOSTS[@]}"; do
  ssh -o ConnectTimeout=6 -o BatchMode=yes "$h" \
    "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits" \
    > "$tmpdir/$h" 2>/dev/null &
done
wait

printf "%-12s %-4s %10s %6s   %s\n" "HOST" "GPU" "FREE(MiB)" "UTIL" "STATUS"
printf -- "------------------------------------------------------------\n"
for h in "${HOSTS[@]}"; do
  if [ -s "$tmpdir/$h" ]; then
    while IFS=',' read -r idx used total util; do
      idx=$(echo "$idx" | xargs); used=$(echo "$used" | xargs)
      total=$(echo "$total" | xargs); util=$(echo "$util" | xargs)
      free=$((total - used))
      if [ "$h" = "a6000-left" ] && [ "$idx" = "0" ]; then
        status="RESERVED (fishonet)"
      elif [ "$used" -lt 2000 ] && [ "$util" -lt 10 ]; then
        status="FREE"
      else
        status="busy"
      fi
      printf "%-12s %-4s %10s %5s%%   %s\n" "$h" "$idx" "$free" "$util" "$status"
    done < "$tmpdir/$h"
  else
    printf "%-12s %-4s %10s %6s   %s\n" "$h" "-" "-" "-" "UNREACHABLE"
  fi
done
rm -rf "$tmpdir"
