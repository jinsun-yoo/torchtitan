#!/bin/bash
#SBATCH --job-name=inprocess_rank_drop
#SBATCH --ntasks=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --partition=champollion
#SBATCH --output=outputs/%j/sbatch.out
#SBATCH --time=00:45:00

set -euxo pipefail

ROOT_PATH="${ROOT_PATH:?}"
SCRIPT_DIR="${ROOT_PATH}/torchtitan"
export RDZV_MASTER_HOSTNAME="$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)"

export NNODES=1
export NRANK_PER_NODE=8
export JOB_CONFIG_NAME="${JOB_CONFIG_NAME:-1b_dp_8_rfsdp}"
export DROP_STEP="${DROP_STEP:-200}"
export DROP_RANK="${DROP_RANK:-1}"
export TOTAL_STEPS="${TOTAL_STEPS:-500}"
export DUMP_FOLDER="${DUMP_FOLDER:-./outputs/rfsdp/1b_dp_8_inprocess}"
# 1B model is used for replicate mode to avoid OOM: 8 full copies on A100 80GB.
# The 8B model OOMs under pure DP-replicate (all weights replicated per rank).

cd "${SCRIPT_DIR}"
source "${ROOT_PATH}/venv/genie/bin/activate"

# Use the standard llama3 model (composable DDP via apply_ddp) not simple_fsdp,
# so the in-place _update_process_group swap works without any model rebuild.
CONFIG_FILE="${SCRIPT_DIR}/torchtitan/models/llama3/train_configs/llama3_${JOB_CONFIG_NAME}.toml" \
TRAIN_FILE="torchtitan.train" \
MODEL_FLAVOR="llama3" \
./run_train.sh \
  --training.seed 42 \
  --training.deterministic \
  --training.steps "${TOTAL_STEPS}" \
  --job.dump_folder "${DUMP_FOLDER}" \
  --checkpoint.no_enable \
  --parallelism.data_parallel_replicate_degree 8 \
  --parallelism.data_parallel_shard_degree 1
#  --training.inprocess_rank_drop_enable \
#  --training.inprocess_rank_drop_step "${DROP_STEP}" \
#  --training.inprocess_rank_to_drop "${DROP_RANK}"
