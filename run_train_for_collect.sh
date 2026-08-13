set -ex
# Set envvars that need to be resolved within this script
export NNODES=${NNODES:-"4"}
DIRPATH="${OUTPUT_PATH}/profile_trace"
if [[ -d ${DIRPATH} ]]; then
  echo "Job output directory ${DIRPATH} already exists. Exiting to avoid overwriting."
  exit 1
fi
mkdir -p ${DIRPATH}

# Set HPE/Genie specific variables/configs here.
export NCCL_NVML_DISABLE=1

if [[ $(hostname) == *sith* ]]; then
  export NCCL_IB_HCA=mlx5_0
  export NCCL_IB_ADAPTIVE_ROUTING=0
fi

# One MPI task per node; torchrun handles NRANK_PER_NODE processes per node internally
# Make sure NRANK_PER_NODE is set before launching mpirun
export NRANK_PER_NODE=${NRANK_PER_NODE:-"1"}
export RDZV_ID=${RDZV_ID:-${SLURM_JOB_ID:-$((RANDOM * 1000 + RANDOM))}}
mpirun -np "$NNODES" -N 1 bw-start ${OUTPUT_PATH}

mpirun -np "$NNODES" -N 1 \
  run_train.sh > \
  ${OUTPUT_PATH}/collect_${JOB_CONFIG_NAME}_iter${ITERATION}.log 2>&1

mpirun -np "$NNODES" -N 1 bw-stop

# Postprocess
python dedup_comm_groups.py ${DIRPATH}

# --gpus=$NRANK_PER_NODE \
