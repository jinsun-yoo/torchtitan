#!/bin/bash
#SBATCH --job-name=collect_chakra
#SBATCH --ntasks=4
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --partition=krai
#SBATCH --account=krai_users
#SBATCH --reservation=HPCDC-29580
#SBATCH --comment='minicol;apm'
#SBATCH --output=outputs/%j/sbatch.out

set -x
squeue -p krai

ROOT_PATH="${ROOT_PATH:?}"
SCRIPT_DIR="${ROOT_PATH}/torchtitan"
OUTPUT_PATH="${SCRIPT_DIR}/outputs/${SLURM_JOB_ID}"
export RDZV_MASTER_HOSTNAME="sith5"
export JOB_CONFIG_NAME="70b_tp_8_fsdp_4"
export NRANK_PER_NODE=8
cd ${SCRIPT_DIR}
bw-start
source ${ROOT_PATH}/venv/genie/bin/activate
CONFIG_FILE="${SCRIPT_DIR}/torchtitan/models/llama3/train_configs/llama3_${JOB_CONFIG_NAME}.toml" bash collect_trace.sh

source ${ROOT_PATH}/venv/hanjiang_chakra/bin/activate
bw-stop
cp *.csv outputs/${SLURM_JOB_ID}/bw_watch.csv