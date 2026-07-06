#!/bin/bash
#SBATCH --job-name=rfsdp_two_phase_rank_drop
#SBATCH --ntasks=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --partition=champollion
#SBATCH --output=outputs/%j/sbatch.out
#SBATCH --time=01:00:00

set -euxo pipefail

ROOT_PATH="${ROOT_PATH:?}"
SCRIPT_DIR="${ROOT_PATH}/torchtitan"
OUTPUT_PATH="${SCRIPT_DIR}/outputs/${SLURM_JOB_ID}"

export RDZV_MASTER_HOSTNAME="$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)"

# Two-phase rank-drop defaults. Override via sbatch --export if needed.
export DROP_STEP="${DROP_STEP:-200}"
export DROP_RANK="${DROP_RANK:-1}"

export PHASE1_NNODES="${PHASE1_NNODES:-1}"
export PHASE1_NPROC_PER_NODE="${PHASE1_NPROC_PER_NODE:-8}"

export PHASE2_NNODES="${PHASE2_NNODES:-1}"
export PHASE2_NPROC_PER_NODE="${PHASE2_NPROC_PER_NODE:-7}"

# Keep local microbatch fixed and allow smaller effective global batch in phase-2.
export PHASE2_GLOBAL_BATCH_SIZE="${PHASE2_GLOBAL_BATCH_SIZE:--1}"
export PHASE2_DELETE_CHECKPOINTS_ON_COMPLETE="${PHASE2_DELETE_CHECKPOINTS_ON_COMPLETE:-true}"

export JOB_CONFIG_NAME="${JOB_CONFIG_NAME:-8b_fsdp_8_rfsdp}"
export CONFIG_FILE="${CONFIG_FILE:-${SCRIPT_DIR}/torchtitan/models/llama3/train_configs/llama3_${JOB_CONFIG_NAME}.toml}"

cd "${SCRIPT_DIR}"
mkdir -p "${OUTPUT_PATH}"

source "${ROOT_PATH}/venv/genie/bin/activate"

# Optional extra args passed to both phases, e.g.
# export EXTRA_ARGS="--training.steps 500 --checkpoint.interval 50"
export EXTRA_ARGS="--training.seed 42 --training.deterministic"
export EXTRA_ARGS="${EXTRA_ARGS:-}"

bash scripts/run_two_phase_rank_drop.sh
