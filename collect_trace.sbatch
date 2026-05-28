#!/bin/bash
#SBATCH --job-name=collect_chakra
#SBATCH --ntasks=8
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --partition=champollion
#SBATCH --comment='minicol;apm'
#SBATCH --output=outputs/%j/sbatch.out
#SBATCH --time=00:30:00

set -x

ROOT_PATH="${ROOT_PATH:?}"
SCRIPT_DIR="${ROOT_PATH}/torchtitan"
OUTPUT_PATH="${SCRIPT_DIR}/outputs/${SLURM_JOB_ID}"
CLUSTER_NAME="champollion"
export RDZV_MASTER_HOSTNAME="$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)"
export JOB_CONFIG_NAME="8b_tp_4_dp_2"
export NRANK_PER_NODE=4
export NNODES=2
export NUM_RANKS=$((NRANK_PER_NODE * NNODES))
export DIRPATH="${SCRIPT_DIR}/outputs/${CLUSTER_NAME}/${JOB_CONFIG_NAME}/profile_trace"
rm -rf ${DIRPATH}
cd ${SCRIPT_DIR}

bw-start
source ${ROOT_PATH}/venv/genie/bin/activate
CONFIG_FILE="${SCRIPT_DIR}/torchtitan/models/llama3/train_configs/llama3_${JOB_CONFIG_NAME}.toml" bash collect_trace.sh
# CONFIG_FILE="${SCRIPT_DIR}/torchtitan/models/deepseek_v3/train_configs/deepseek_v3_${JOB_CONFIG_NAME}.toml" bash collect_trace.sh

bw-stop
cp *.csv outputs/${SLURM_JOB_ID}/bw_watch.csv
bash combine_trace.sh

python dedup_comm_groups.py ${JOB_CONFIG_NAME} ${CLUSTER_NAME}