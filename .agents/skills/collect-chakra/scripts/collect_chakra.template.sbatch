#!/bin/bash
#SBATCH --job-name=collect_chakra
#SBATCH --ntasks={{AGENT_NRANKS}}
#SBATCH --nodes={{AGENT_NNODES}}
#SBATCH --ntasks-per-node={{AGENT_NRANK_PER_NODE}}
#SBATCH --partition={{AGENT_PARTITION}}
#SBATCH --output=outputs/%j/sbatch.out
#SBATCH --time={{AGENT_TIME_LIMIT}}

set -euxo pipefail

# Cluster wide setting
ROOT_PATH="${ROOT_PATH:?ROOT_PATH must be set}"
SCRIPT_DIR="${ROOT_PATH}/torchtitan"
CLUSTER_NAME="{{AGENT_PARTITION}}"

# Per slurm-job setting
RDZV_MASTER_HOSTNAME="$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)"
NRANK_PER_NODE={{AGENT_NRANK_PER_NODE}}
NNODES={{AGENT_NNODES}}
NUM_RANKS={{AGENT_NRANKS}}

cd "${SCRIPT_DIR}"
source "${ROOT_PATH}/venv/genie/bin/activate"

export ROOT_PATH SCRIPT_DIR NRANK_PER_NODE NNODES NUM_RANKS RDZV_MASTER_HOSTNAME
for JOB_CONFIG_NAME in {{AGENT_JOBCONFIGS}}; do
  for ITERATION in $(seq {{AGENT_ITERATION_START}} {{AGENT_ITERATION_END}}); do
    JOB_OUTPUT_DIR="${SCRIPT_DIR}/outputs/{{AGENT_PARTITION}}/${JOB_CONFIG_NAME}/${ITERATION}"
    DIRPATH="${JOB_OUTPUT_DIR}/profile_trace"
    MODEL_NAME="${MODEL_NAME:?MODEL_NAME must be set}"
    MODEL_FLAVOR="${MODEL_FLAVOR:-simple_fsdp.llama3}"

    export JOB_CONFIG_NAME MODEL_NAME MODEL_FLAVOR ITERATION JOB_OUTPUT_DIR DIRPATH
    bash single_config.sh
  done
done
