set -x
RDZV_MASTER_HOSTNAME=${RDZV_MASTER_HOSTNAME:-"g100n052"}
MODEL_NAME=${MODEL_NAME:-"llama3"}
JOB_CONFIG_NAME=${JOB_CONFIG_NAME:-"1b_dp_4"}
CONFIG_FILE_DEFAULT="./torchtitan/models/${MODEL_NAME}/train_configs/${MODEL_NAME}_${JOB_CONFIG_NAME}.toml"
CONFIG_FILE=${CONFIG_FILE:-$CONFIG_FILE_DEFAULT}
NNODES=${NNODES:-"4"}
NRANK_PER_NODE=${NRANK_PER_NODE:-"1"}

export CONFIG_FILE=$CONFIG_FILE
export RDZV_MASTER_HOSTNAME=$RDZV_MASTER_HOSTNAME
export CUDA_VISIBLE_DEVICES=0
export NNODES=$NNODES
export NRANK_PER_NODE=$NRANK_PER_NODE
export MODEL_NAME=$MODEL_NAME

MCA_STRING=""
if [[ $(hostname) == *sith* ]]; then
  export NCCL_IB_HCA=mlx5_0
  export NCCL_IB_ADAPTIVE_ROUTING=0
  # This option is needed to run mpirun in vader nodes where mpirun clashes with srun.
  MCA_STRING="--mca btl_tcp_if_include bond0"
fi

mpirun $MCA_STRING -np $((NNODES * NRANK_PER_NODE)) -N $NRANK_PER_NODE -x CUDA_VISIBLE_DEVICES -x NCCL_NVML_DISABLE run_train.sh > \
collect_${JOB_CONFIG_NAME}.log 2>&1

