#!/bin/bash
#SBATCH --job-name=collect_chakra_champollion_8b_fsdp_4_multiiter
#SBATCH --ntasks=4
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --partition=champollion
#SBATCH --comment='minicol;apm'
#SBATCH --output=outputs/%j/sbatch.out

set -x
squeue -p champollion

SCRIPT_DIR="/nfs/jinsun/torchtitan"
OUTPUT_PATH="${SCRIPT_DIR}/outputs/${SLURM_JOB_ID}"

cd /nfs/jinsun/torchtitan
bw-start
source /nfs/jinsun/pythorch/bin/activate
CONFIG_FILE="./torchtitan/models/llama3/train_configs/llama3_8b_tp_2_multi.toml" bash collect_trace.sh

source /nfs/jinsun/venv/hanjiang_chakra/bin/activate
bw-stop
cp *.csv outputs/${SLURM_JOB_ID}/bw_watch.csv