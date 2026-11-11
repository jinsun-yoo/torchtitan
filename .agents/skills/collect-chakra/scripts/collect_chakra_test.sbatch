#!/bin/bash
#SBATCH --job-name=collect_chakra
#SBATCH --ntasks=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --partition=champollion
#SBATCH --output=outputs/%j/sbatch.out
#SBATCH --time=00:30:00

set -euxo pipefail

# Cluster wide setting
ROOT_PATH="${ROOT_PATH:?ROOT_PATH must be set}"
SCRIPT_DIR="${ROOT_PATH}/torchtitan"
CLUSTER_NAME="champollion"

# Per slurm-job setting
RDZV_MASTER_HOSTNAME="$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)"
NRANK_PER_NODE=8
NNODES=1
NUM_RANKS=8

cd "${SCRIPT_DIR}"
source "${ROOT_PATH}/venv/genie/bin/activate"

export ROOT_PATH SCRIPT_DIR NRANK_PER_NODE NNODES NUM_RANKS RDZV_MASTER_HOSTNAME
for JOB_CONFIG_NAME in 8b_tp_8; do
  for ITERATION in $(seq 4 4); do
    JOB_OUTPUT_DIR="${SCRIPT_DIR}/outputs/champollion/${JOB_CONFIG_NAME}/${ITERATION}"
    DIRPATH="${JOB_OUTPUT_DIR}/profile_trace"
    MODEL_NAME="${MODEL_NAME:?MODEL_NAME must be set}"
    MODEL_FLAVOR="${MODEL_FLAVOR:-simple_fsdp.llama3}"

    export JOB_CONFIG_NAME MODEL_NAME MODEL_FLAVOR ITERATION JOB_OUTPUT_DIR DIRPATH
    bash single_config.sh
  done
done
