#!/usr/bin/env bash
set -euo pipefail
set -x

# Serial ablation launcher:
#   1) train SpatialVLA + FIFO memory + LoRA
#   2) clean up torchrun/python workers and wait for GPU memory to be released
#   3) train SpatialVLA + LoRA without memory using matching defaults
#
# Smoke test example:
#   MAX_STEPS=2 FIX_RAW_LENGTH=16 epoch=1 save_steps=1000 bash scripts/spatialvla_4b_finetune/train_memory_then_lora.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_FREE_THRESHOLD_MB=${GPU_FREE_THRESHOLD_MB:-2000}
GPU_FREE_WAIT_SECONDS=${GPU_FREE_WAIT_SECONDS:-600}

cleanup_training_processes() {
  pkill -f "python.*train/spatialvla_finetune.py" || true
  pkill -f "torchrun.*train/spatialvla_finetune.py" || true
}

wait_for_gpu_memory() {
  local deadline=$((SECONDS + GPU_FREE_WAIT_SECONDS))
  local used_mb

  while [ "${SECONDS}" -lt "${deadline}" ]; do
    used_mb=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{sum += $1} END {print sum + 0}')
    if [ "${used_mb}" -le "${GPU_FREE_THRESHOLD_MB}" ]; then
      nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
      return 0
    fi
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
    sleep 10
  done

  echo "GPU memory did not fall below ${GPU_FREE_THRESHOLD_MB} MB within ${GPU_FREE_WAIT_SECONDS}s." >&2
  return 1
}

run_training_stage() {
  local stage_name="$1"
  local script_path="$2"
  local exit_code=0

  echo "===== START ${stage_name} ====="
  bash "${script_path}" || exit_code=$?
  echo "===== CLEANUP ${stage_name} ====="
  cleanup_training_processes
  sleep 20
  wait_for_gpu_memory

  if [ "${exit_code}" -ne 0 ]; then
    echo "${stage_name} failed with exit code ${exit_code}." >&2
    return "${exit_code}"
  fi
  echo "===== DONE ${stage_name} ====="
}

trap cleanup_training_processes EXIT

run_training_stage "memory_lora" "${SCRIPT_DIR}/finetune_memory_lora.sh"
run_training_stage "no_memory_lora" "${SCRIPT_DIR}/finetune_lora.sh"
