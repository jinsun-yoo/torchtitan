set -ex
# Set envvars that need to be resolved within this script
export NNODES=${NNODES:-"4"}
export MODEL_NAME=${MODEL_NAME:-"llama3"}
export JOB_CONFIG_NAME=${JOB_CONFIG_NAME:-"1b_dp_4"}
export ITERATION=${ITERATION:-"0"}
export JOB_OUTPUT_DIR="${SCRIPT_DIR}/outputs/${CLUSTER_NAME}/${JOB_CONFIG_NAME}/${ITERATION}"
export PROFILE_TRACE_DIR="${JOB_OUTPUT_DIR}/profile_trace"
export DIRPATH="${PROFILE_TRACE_DIR}"
if [[ -d ${JOB_OUTPUT_DIR} ]]; then
  echo "Job output directory ${JOB_OUTPUT_DIR} already exists. Exiting to avoid overwriting."
  exit 1
fi
mkdir -p ${JOB_OUTPUT_DIR}

# Set HPE/Genie specific variables/configs here.
export CONFIG_FILE=${CONFIG_FILE:-"./torchtitan/models/${MODEL_NAME}/train_configs/${MODEL_NAME}_${JOB_CONFIG_NAME}.toml"}
export NCCL_NVML_DISABLE=1

if [[ $(hostname) == *sith* ]]; then
  export NCCL_IB_HCA=mlx5_0
  export NCCL_IB_ADAPTIVE_ROUTING=0
fi

# Generate RDZV_ID once for all tasks to share
export RDZV_ID=$((RANDOM * 1000 + RANDOM))


# One srun task per node; torchrun handles NRANK_PER_NODE processes per node internally
bw-start
srun \
  --nodes=$NNODES \
  --ntasks=$NNODES \
  --ntasks-per-node=1 \
  --distribution=block:block \
  --export=ALL \
  run_train.sh > \
${JOB_OUTPUT_DIR}/collect_${JOB_CONFIG_NAME}_iter${ITERATION}.log 2>&1
bw-stop

# Postprocess
latest_csv="$(ls -1t bwmonitor-*.csv 2>/dev/null | head -n 1 || true)"
if [[ -n "${latest_csv}" ]]; then
  mv "${latest_csv}" "${JOB_OUTPUT_DIR}/bw_watch.csv"
fi
python dedup_comm_groups.py ${JOB_OUTPUT_DIR}

# --gpus=$NRANK_PER_NODE \
