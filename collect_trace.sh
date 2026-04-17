set -x
RDZV_MASTER_HOSTNAME=${RDZV_MASTER_HOSTNAME:-"g100n052"}
MODEL_NAME=${MODEL_NAME:-"llama3"}
JOB_CONFIG_NAME=${JOB_CONFIG_NAME:-"1b_dp_4"}
CONFIG_FILE_DEFAULT="./torchtitan/models/${MODEL_NAME}/train_configs/${MODEL_NAME}_${JOB_CONFIG_NAME}.toml"
CONFIG_FILE=${CONFIG_FILE:-$CONFIG_FILE_DEFAULT}
NNODES=${NNODES:-"4"}
NRANK_PER_NODE=${NRANK_PER_NODE:-"1"}
NTASKS=$((NNODES * NRANK_PER_NODE))

export CONFIG_FILE=$CONFIG_FILE
export RDZV_MASTER_HOSTNAME=$RDZV_MASTER_HOSTNAME
export NNODES=$NNODES
export NRANK_PER_NODE=$NRANK_PER_NODE

export NCCL_NVML_DISABLE=1

if [[ $(hostname) == *sith* ]]; then
  export NCCL_IB_HCA=mlx5_0
  export NCCL_IB_ADAPTIVE_ROUTING=0
fi

# Generate RDZV_ID once for all tasks to share
export RDZV_ID=$((RANDOM * 1000 + RANDOM))

# --ntasks=$NTASKS \
# --nodes=$NNODES \
# --ntasks-per-node=$NRANK_PER_NODE \
srun \
  --ntasks=$NNODES \
  --ntasks-per-node=1 \
  --export=ALL \
  run_train.sh > \
collect_${JOB_CONFIG_NAME}.log 2>&1

# --gpus=$NRANK_PER_NODE \
