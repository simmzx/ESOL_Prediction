#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${DATA_PATH:-/home/zhangxiang_cluster/assign1/esol.csv}"
OUT="${OUT_DIR:-$ROOT/results/run_$(date +%Y%m%d_%H%M%S)}"
SPLIT="${SPLIT:-random}"
SEED="${SEED:-42}"
THREADS="${THREADS:-4}"
RUN_MODE="${RUN_MODE:-standard}"
XGB_DEVICE="${XGB_DEVICE:-cpu}"
REQUIRE_GPU="${REQUIRE_GPU:-1}"
GPU_INDEX="${GPU_INDEX:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
while (($#)); do
 case "$1" in
   --data) DATA="$2";shift 2;;
   --output) OUT="$2";shift 2;;
   --split) SPLIT="$2";shift 2;;
   --seed) SEED="$2";shift 2;;
   --mode) RUN_MODE="$2";shift 2;;
   --gpu) GPU_INDEX="$2";shift 2;;
   --threads) THREADS="$2";shift 2;;
   --xgb-device) XGB_DEVICE="$2";shift 2;;
   --allow-cpu) REQUIRE_GPU=0;shift;;
   *) echo "Unknown argument: $1" >&2;exit 2;;
 esac
done
[[ -f "$DATA" ]] || { echo "Cannot find CSV: $DATA" >&2;exit 3; }
[[ "$SPLIT" == random || "$SPLIT" == scaffold ]] || { echo 'split must be random/scaffold' >&2;exit 2; }
[[ "$RUN_MODE" == fast || "$RUN_MODE" == standard ]] || { echo 'mode must be fast/standard' >&2;exit 2; }
[[ "$XGB_DEVICE" == cpu || "$XGB_DEVICE" == cuda ]] || { echo 'xgb-device must be cpu/cuda' >&2;exit 2; }
mkdir -p "$OUT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" NUMEXPR_NUM_THREADS="$THREADS"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
exec > >(tee -a "$OUT/pipeline.log") 2>&1
echo "[START] $(date -Is)"
echo "[CONFIG] root=$ROOT data=$DATA output=$OUT split=$SPLIT seed=$SEED mode=$RUN_MODE gpu_index=$GPU_INDEX"
echo "[GPU] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (torch sees physical GPU $GPU_INDEX as cuda:0)"
if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi -i "$GPU_INDEX" --query-gpu=index,name,memory.total --format=csv,noheader || true;fi
"$PYTHON_BIN" "$ROOT/scripts/check_env.py" $( [[ "$REQUIRE_GPU" == 1 ]] && echo '--require-gpu' )
"$PYTHON_BIN" -m src.prepare --data "$DATA" --output "$OUT" --split "$SPLIT" --seed "$SEED" --fp-bits 1024
# xgboost on CPU is often faster for this 1128-row task; optional XGB_DEVICE=cuda uses CUDA_VISIBLE_DEVICES.
"$PYTHON_BIN" -m src.train_ml --input "$OUT" --threads "$THREADS" --xgb-device "$XGB_DEVICE"
RNN_ARGS=(--input "$OUT" --architecture both --teacher-epochs 30 --attention-epochs 80 --patience 12 --batch-size 32)
if [[ "$RUN_MODE" == standard ]]; then RNN_ARGS+=(--seeds 42 43 44);else RNN_ARGS+=(--seeds 42);fi
if [[ "$REQUIRE_GPU" == 1 ]];then RNN_ARGS+=(--require-gpu);fi
"$PYTHON_BIN" -m src.train_rnn "${RNN_ARGS[@]}"
"$PYTHON_BIN" -m src.summarize --input "$OUT"
echo "[SUCCESS] $(date -Is) result folder: $OUT"
