set -euo pipefail

SCRIPT_DIR="${SCRIPT_DIR:?SCRIPT_DIR must be set}"
CLUSTER_NAME="${CLUSTER_NAME:?CLUSTER_NAME must be set}"
JOB_CONFIG_NAME="${JOB_CONFIG_NAME:?JOB_CONFIG_NAME must be set}"
ITERATION="${ITERATION:?ITERATION must be set}"

bash run_train_for_collect.sh

DIRPATH="${SCRIPT_DIR}/outputs/${CLUSTER_NAME}/${JOB_CONFIG_NAME}/${ITERATION}/profile_trace" \
NUM_RANKS="${NUM_RANKS:?NUM_RANKS must be set}" \
bash combine_trace.sh