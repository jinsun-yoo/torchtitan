#!/bin/bash
set -ex

NNODES=${NNODES:-"1"}
if [[ "${NNODES}" != "1" ]]; then
  echo "run_train_for_collect_singlenode.sh only supports NNODES=1, got ${NNODES}" >&2
  exit 1
fi

PROFILE_TRACE_PATH="${OUTPUT_PATH}/profile_trace"
if [[ -d ${PROFILE_TRACE_PATH} ]]; then
  echo "Job output directory ${PROFILE_TRACE_PATH} already exists. Exiting to avoid overwriting."
  exit 1
fi

NRANKS=${NRANKS:-"4"}
NRANK_PER_NODE=${NRANK_PER_NODE:-"1"}
RDZV_ID=${RDZV_ID:-${SLURM_JOB_ID:-$((RANDOM * 1000 + RANDOM))}}
TRAIN_FILE=${TRAIN_FILE:-"torchtitan.train"}
SEED_HARDCODE=42

RDZV_MASTER_HOSTNAME=${RDZV_MASTER_HOSTNAME:-"localhost"}
export TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"$RDZV_MASTER_HOSTNAME:29510"}

export NCCL_NVML_DISABLE=1
if [[ $(hostname) == *sith* ]]; then
  export NCCL_IB_HCAS=${NCCL_IB_HCAS:-"mlx5_0,mlx5_2,mlx5_8,mlx5_10"}
  export NCCL_IB_ADAPTIVE_ROUTING=${NCCL_IB_ADAPTIVE_ROUTING:-0}
fi
export PYTORCH_ALLOC_CONF="expandable_segments:True"
CONFIG_FILE=${CONFIG_FILE:-"${ROOT_PATH}/torchtitan/torchtitan/models/llama3/train_configs/debug_model.toml"}

export TRAIN_FILE CONFIG_FILE SEED_HARDCODE OUTPUT_PATH MODEL_NAME ENABLE_MEMORY_SNAPSHOT
export NCCL_IB_HCAS

torchrun --standalone --nproc-per-node="${NRANK_PER_NODE}" --no-python \
  bash -lc '
    set -euo pipefail

    if [[ -n "${NCCL_IB_HCAS:-}" ]]; then
      local_rank="${LOCAL_RANK:-0}"
      IFS=, read -r -a hcas <<< "${NCCL_IB_HCAS}"

      if (( local_rank >= ${#hcas[@]} )); then
        echo "Missing NCCL_IB_HCA entry for local rank ${local_rank}: NCCL_IB_HCAS=${NCCL_IB_HCAS}" >&2
        exit 1
      fi

      export NCCL_IB_HCA="${hcas[$local_rank]}"
    fi

    cmd=(
      python -m "${TRAIN_FILE}"
      --job.config_file "${CONFIG_FILE}"
      --training.seed "${SEED_HARDCODE}"
      --training.deterministic
    )

    if [[ -n "${MODEL_NAME:-}" ]]; then
      cmd+=(--model.name "${MODEL_NAME}")
    fi

    if [[ -n "${OUTPUT_PATH:-}" ]]; then
      cmd+=(--job.dump_folder "${OUTPUT_PATH}")
    fi

    if [[ -n "${ENABLE_MEMORY_SNAPSHOT:-}" ]]; then
      cmd+=(--profiling.enable_memory_snapshot "${ENABLE_MEMORY_SNAPSHOT}")
    fi

    cmd+=("$@")
    exec "${cmd[@]}"
  ' bash "$@" \
  > "${OUTPUT_PATH}/collect_${JOB_CONFIG_NAME}_iter${ITERATION}.log" 2>&1