#!/bin/bash
#SBATCH --job-name=collect_chakra
#SBATCH --ntasks=4
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --comment='minicol;apm'
#SBATCH --output=outputs/%j/sbatch.out

set -x
squeue -p krai

ROOT_PATH="${ROOT_PATH:?}"
SCRIPT_DIR="${ROOT_PATH}/torchtitan"
OUTPUT_PATH="${SCRIPT_DIR}/outputs/${SLURM_JOB_ID}"
export RDZV_MASTER_HOSTNAME="$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)"
export JOB_CONFIG_NAME="16b_ep4"
export NRANK_PER_NODE=1
export NNODES=4
cd ${SCRIPT_DIR}
bw-start
source ${ROOT_PATH}/venv/genie/bin/activate
# CONFIG_FILE="${SCRIPT_DIR}/torchtitan/models/llama3/train_configs/llama3_${JOB_CONFIG_NAME}.toml" bash collect_trace.sh
CONFIG_FILE="${SCRIPT_DIR}/torchtitan/models/deepseek_v3/train_configs/deepseek_v3_${JOB_CONFIG_NAME}.toml" bash collect_trace.sh

bw-stop
cp *.csv outputs/${SLURM_JOB_ID}/bw_watch.csv