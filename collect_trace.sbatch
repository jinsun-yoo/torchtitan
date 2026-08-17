#!/bin/bash
#SBATCH --job-name=collect_chakra
#SBATCH --ntasks=64
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --partition=champollion
#SBATCH --comment='minicol;apm'
#SBATCH --output=/nfs/jinsun/rfsdp_root/outputs/%j/sbatch.out
#SBATCH --time=00:30:00

set -euxo pipefail

ROOT_PATH="${ROOT_PATH:?}"
export SCRIPT_DIR="${ROOT_PATH}/torchtitan"
STDOUT_PATH=$(scontrol show job -o "$SLURM_JOB_ID" \
	| tr ' ' '\n' \
	| awk -F= '$1=="StdOut"{print $2}')
export OUTPUT_PATH
OUTPUT_PATH=$(dirname "$STDOUT_PATH")
export CLUSTER_NAME="champollion"
export RDZV_MASTER_HOSTNAME="$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)"
export JOB_CONFIG_NAME="8b_fsdp_64"
export JOB_OUTPUT_DIR="${OUTPUT_PATH}/${JOB_CONFIG_NAME}"
export NRANK_PER_NODE=8
export NNODES=8
export NUM_RANKS=$((NRANK_PER_NODE * NNODES))
cd "${SCRIPT_DIR}"

source "${ROOT_PATH}/venv/rfsdp/bin/activate"
CONFIG_FILE="${SCRIPT_DIR}/torchtitan/models/llama3/train_configs/llama3_${JOB_CONFIG_NAME}.toml" bash run_train_for_collect.sh
# CONFIG_FILE="${SCRIPT_DIR}/torchtitan/models/deepseek_v3/train_configs/deepseek_v3_${JOB_CONFIG_NAME}.toml" bash collect_trace.sh

compgen -G "${JOB_OUTPUT_DIR}/profile_trace/*.json" > /dev/null