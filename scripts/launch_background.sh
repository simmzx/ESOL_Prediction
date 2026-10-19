#!/usr/bin/env bash
# Starts a detached single experiment, prints PID/log path, does not install packages.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${OUT_DIR:-$ROOT/results/exp_${STAMP}}"
ARGS=("$@")
# If caller passes --output, honor it, so print and PID paths agree.
for ((i=0;i<${#ARGS[@]};i++));do
 if [[ "${ARGS[$i]}" == '--output' ]];then OUT="${ARGS[$((i+1))]}";fi
done
mkdir -p "$OUT"
nohup bash "$ROOT/scripts/run_all.sh" --output "$OUT" "${ARGS[@]}" > "$OUT/nohup.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$OUT/launcher.pid"
echo "BACKGROUND PID=$pid"
echo "OUTPUT=$OUT"
echo "FOLLOW: tail -f '$OUT/nohup.log'"
echo "STATUS: ps -fp $pid"
