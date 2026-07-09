#!/bin/bash
# ponytail: reuses vla_dashboard.sh's exact SSH queries; renders into
# n-vla-dashboard.html instead of the terminal. --loop keeps it fresh while
# the page (which auto-reloads itself) is open in a browser.
set -e
HOST="a6000-left"
REMOTE_DIR='C:\Users\islab01\vla-atlas\experiments\firstpass_sweep'
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEMPLATE="$DIR/n-vla-dashboard.template.html"
OUT="$DIR/n-vla-dashboard.html"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

LOOP=0
INTERVAL=15
if [ "$1" = "--loop" ]; then
  LOOP=1
  INTERVAL="${2:-15}"
fi

render_once() {
  date '+%Y-%m-%d %H:%M:%S' > "$WORK/ts.txt"

  ssh -o ConnectTimeout=8 -o BatchMode=yes "$HOST" \
    "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits" \
    > "$WORK/gpu.csv" 2>/dev/null || true

  ssh -o ConnectTimeout=8 -o BatchMode=yes "$HOST" "tasklist | findstr /i python.exe" \
    > "$WORK/proc.txt" 2>/dev/null || true

  ssh -o ConnectTimeout=8 -o BatchMode=yes "$HOST" powershell -NoProfile -Command \
    "\"Get-ChildItem -Path '$REMOTE_DIR\\decoder_fix_validation*.json','$REMOTE_DIR\\results.json' -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty Name\"" \
    2>/dev/null | tr -d '\r' > "$WORK/newest_name.txt" || true

  newest="$(cat "$WORK/newest_name.txt")"
  if [[ "$newest" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    ssh -o ConnectTimeout=8 -o BatchMode=yes "$HOST" \
      "type $REMOTE_DIR\\$newest" \
      > "$WORK/results.json" 2>/dev/null || echo "[]" > "$WORK/results.json"
  else
    echo "none" > "$WORK/newest_name.txt"
    echo "[]" > "$WORK/results.json"
  fi

  python "$DIR/scripts/render_dashboard.py" "$TEMPLATE" "$OUT" "$WORK" "$INTERVAL"
}

if [ "$LOOP" = "1" ]; then
  echo "Polling $HOST every ${INTERVAL}s -> $OUT (Ctrl+C to stop)"
  while true; do
    render_once
    echo "[$(date '+%H:%M:%S')] refreshed"
    sleep "$INTERVAL"
  done
else
  render_once
  echo "Wrote $OUT"
fi
