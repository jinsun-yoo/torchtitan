set -ex
# Set envvars that need to be resolved within this script
NNODES=${NNODES:-"4"}
PROFILE_TRACE_PATH="${OUTPUT_PATH}/profile_trace"
if [[ -d ${PROFILE_TRACE_PATH} ]]; then
  echo "Job output directory ${PROFILE_TRACE_PATH} already exists. Exiting to avoid overwriting."
  exit 1
fi

# One MPI task per node; torchrun handles NRANK_PER_NODE processes per node internally
# Make sure NRANK_PER_NODE is set before launching mpirun
NRANKS=${NRANKS:-"4"}
NRANK_PER_NODE=${NRANK_PER_NODE:-"1"}
RDZV_ID=${RDZV_ID:-${SLURM_JOB_ID:-$((RANDOM * 1000 + RANDOM))}}
TRAIN_FILE=${TRAIN_FILE:-"torchtitan.train"}
SEED_HARDCODE=42

# For single-node runs, use localhost for rendezvous to avoid cross-node connection issues
RDZV_MASTER_HOSTNAME=${RDZV_MASTER_HOSTNAME:-"g100n052"}
export TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"$RDZV_MASTER_HOSTNAME:29510"}
if [ "$NNODES" -eq 1 ]; then
  RDZV_ENDPOINT="localhost:29500"
else
  RDZV_ENDPOINT="$RDZV_MASTER_HOSTNAME:29500"
fi

# Set HPE/Genie specific variables/configs here.
export NCCL_NVML_DISABLE=1
if [[ $(hostname) == *sith* ]]; then
  export NCCL_IB_HCA=mlx5_0
  export NCCL_IB_ADAPTIVE_ROUTING=0
fi
export PYTORCH_ALLOC_CONF="expandable_segments:True"
CONFIG_FILE=${CONFIG_FILE:-"${ROOT_PATH}/torchtitan/torchtitan/models/llama3/train_configs/debug_model.toml"}

# Actual Run
mpirun -np "$NNODES" -N 1 \
  -x PYTORCH_ALLOC_CONF \
  -x TORCHFT_LIGHTHOUSE \
  -x NCCL_NVML_DISABLE \
  ${NCCL_IB_HCA:+-x NCCL_IB_HCA} \
  ${NCCL_IB_ADAPTIVE_ROUTING:+-x NCCL_IB_ADAPTIVE_ROUTING} \
  torchrun --nnodes=${NNODES} --nproc-per-node=${NRANK_PER_NODE} \
    --rdzv-id=$RDZV_ID --rdzv-backend=c10d --rdzv-endpoint=$RDZV_ENDPOINT \
    -m ${TRAIN_FILE} --job.config_file ${CONFIG_FILE} \
    --training.seed ${SEED_HARDCODE} --training.deterministic \
    ${MODEL_NAME:+--model.name ${MODEL_NAME}} \
    ${OUTPUT_PATH:+--job.dump_folder ${OUTPUT_PATH}} \
    ${ENABLE_MEMORY_SNAPSHOT:+--profiling.enable_memory_snapshot ${ENABLE_MEMORY_SNAPSHOT}} \
    "$@" \
  > ${OUTPUT_PATH}/collect_${JOB_CONFIG_NAME}_iter${ITERATION}.log 2>&1

# Postprocess
python dedup_comm_groups.py ${OUTPUT_PATH}

# --gpus=$NRANK_PER_NODE \
