#!/usr/bin/env bash
# GPU monitor across all 6 lab hosts. Replaces scripts/gpu_check.sh + scripts/gpu_dashboard.sh.
#
#   ./gpu.sh              live, refresh every 5s, runs until Ctrl+C (default)
#   ./gpu.sh 15           live, refresh every 15s
#   ./gpu.sh --once       one-shot snapshot, then exit
#   ./gpu.sh --selftest   parser checks against fixtures, no SSH, no GPU
#
# Robustness the old scripts lacked (each one bit us or was one driver away from it):
#   - Windows hosts emit CRLF. The old scripts survived only because GNU xargs
#     happens to strip \r. Stripped explicitly here.
#   - nvidia-smi reports [N/A] for util/memory on MIG and some drivers. That
#     crashed `[ "$used" -lt 2000 ]` and `$((total-used))`. Now reported as UNKNOWN.
#   - `read` silently drops a final line with no trailing newline. Guarded.
#   - "unreachable host" and "host up but nvidia-smi returned nothing" were
#     conflated. Now distinct: UNREACHABLE vs NO-DATA.
#
# ponytail: serial-parallel ssh fan-out, no agent/daemon. Swap for dcgm-exporter
# + Grafana only if this ever needs to scale past a handful of boxes.

# No `set -e`: one dead host must not kill the sweep.
set -uo pipefail

HOSTS=(a100 blackwell a6000 blackwell2 a6000-left a6000-mid)
FREE_MEM_MIB=2000   # below this AND below FREE_UTIL_PCT => nothing running right now
FREE_UTIL_PCT=10

is_num() { case "${1:-}" in '' | *[!0-9]*) return 1 ;; *) return 0 ;; esac; }

# a6000-left GPU0 is permanently reserved for another lab project (fishonet).
is_reserved() { [ "$1" = "a6000-left" ] && [ "$2" = "0" ]; }

# host idx used util -> status string
status_of() {
  local host=$1 idx=$2 used=$3 util=$4
  if is_reserved "$host" "$idx"; then echo "RESERVED (fishonet)"; return; fi
  if ! is_num "$used" || ! is_num "$util"; then echo "UNKNOWN (n/a)"; return; fi
  if [ "$used" -lt "$FREE_MEM_MIB" ] && [ "$util" -lt "$FREE_UTIL_PCT" ]; then
    echo "free*"
  else
    echo "busy"
  fi
}

# Emits "idx|free_mib|util|status" per GPU. Reads CSV on stdin.
# `|| [ -n "$line" ]` keeps the final line when there is no trailing newline.
parse_host() {
  local host=$1 line idx used total util free
  while IFS= read -r line || [ -n "$line" ]; do
    line=${line%$'\r'}                       # Windows hosts emit CRLF
    [ -z "${line// /}" ] && continue
    IFS=',' read -r idx used total util <<<"$line"
    idx=${idx// /}; used=${used// /}; total=${total// /}; util=${util// /}
    if is_num "$used" && is_num "$total"; then free=$((total - used)); else free="-"; fi
    printf '%s|%s|%s|%s\n' "$idx" "$free" "$util" "$(status_of "$host" "$idx" "$used" "$util")"
  done
}

snapshot() {
  local tmpdir h rc
  tmpdir=$(mktemp -d)
  for h in "${HOSTS[@]}"; do
    { ssh -o ConnectTimeout=6 -o BatchMode=yes "$h" \
        "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits" \
        >"$tmpdir/$h" 2>/dev/null; echo $? >"$tmpdir/$h.rc"; } &
  done
  wait

  printf "%-12s %-4s %11s %7s   %s\n" "HOST" "GPU" "FREE(MiB)" "UTIL" "STATUS"
  printf -- "--------------------------------------------------------------\n"
  for h in "${HOSTS[@]}"; do
    rc=$(cat "$tmpdir/$h.rc" 2>/dev/null || echo 255)
    if [ "$rc" != "0" ]; then
      printf "%-12.12s %-4s %11s %7s   %s\n" "$h" "-" "-" "-" "UNREACHABLE (ssh rc=$rc)"
    elif [ ! -s "$tmpdir/$h" ]; then
      printf "%-12.12s %-4s %11s %7s   %s\n" "$h" "-" "-" "-" "NO-DATA (no nvidia-smi?)"
    else
      parse_host "$h" <"$tmpdir/$h" | while IFS='|' read -r idx free util status; do
        printf "%-12.12s %-4s %11s %6s%%   %s\n" "$h" "$idx" "$free" "$util" "$status"
      done
    fi
  done
  rm -rf "$tmpdir"
  echo
  echo "free* = nothing allocated right now. These are SHARED boxes: check the process"
  echo "owner (nvidia-smi / tasklist) before claiming a GPU."
}

selftest() {
  local out
  # CRLF + a normal Linux line, no trailing newline on the last row.
  out=$(printf '0, 611, 49140, 0\r\n1, 12, 49140, 0\r\n' | parse_host a6000-mid)
  [ "$(echo "$out" | wc -l)" = "2" ] || { echo "FAIL: CRLF rows"; return 1; }
  echo "$out" | grep -q '^0|48529|0|free\*$' || { echo "FAIL: CRLF strip/classify: $out"; return 1; }

  # Final line with no trailing newline must not be dropped.
  out=$(printf '0, 100, 1000, 5\n1, 200, 1000, 5' | parse_host a100)
  [ "$(echo "$out" | wc -l)" = "2" ] || { echo "FAIL: dropped last line"; return 1; }

  # [N/A] must not crash arithmetic or the -lt tests.
  out=$(printf '0, [N/A], [N/A], [N/A]\n' | parse_host a100)
  echo "$out" | grep -q 'UNKNOWN' || { echo "FAIL: N/A not handled: $out"; return 1; }
  echo "$out" | grep -q '^0|-|' || { echo "FAIL: N/A free should be '-': $out"; return 1; }

  # Reserved GPU wins over any free/busy reading.
  out=$(printf '0, 5, 49140, 0\n' | parse_host a6000-left)
  echo "$out" | grep -q 'RESERVED' || { echo "FAIL: reserved not flagged: $out"; return 1; }

  # A busy GPU is busy: low memory but high util, and high memory but idle util.
  echo "$(printf '0, 5, 49140, 90\n' | parse_host a100)" | grep -q 'busy' || { echo "FAIL: high-util busy"; return 1; }
  echo "$(printf '0, 40000, 49140, 0\n' | parse_host a100)" | grep -q 'busy' || { echo "FAIL: high-mem busy"; return 1; }

  echo "selftest OK"
}

usage() { sed -n '2,9p' "$0" | sed 's/^# \?//'; }

# Double-clicked from Explorer? The window dies with the shell on any exit path.
# Hold it open so the error or the usage text is actually readable.
# ponytail: only when stdin is a TTY -- otherwise this would hang CI/pipes forever.
hold_open() { [ -t 0 ] && { echo; read -r -n 1 -p "Press any key to close..."; echo; }; }

loop() {
  local interval=$1
  # Ctrl+C exits cleanly instead of dumping a partial frame + trap noise.
  trap 'echo; echo "stopped."; exit 0' INT
  while true; do
    clear
    echo "GPU Dashboard  $(date '+%Y-%m-%d %H:%M:%S')  (refresh ${interval}s, Ctrl+C to quit)"
    echo
    snapshot
    sleep "$interval"
  done
}

case "${1:-}" in
  --selftest) selftest ;;
  --once) snapshot ;;
  --help | -h) usage ;;
  '') loop 5 ;;
  --loop) loop "${2:-5}" ;;                       # kept: old muscle memory / docs
  *[!0-9]*) usage >&2; hold_open; exit 2 ;;       # anything non-numeric is a bad arg
  *) loop "$1" ;;                                 # bare number = interval
esac
