#!/usr/bin/env bash
# Two-phase rank-drop launcher for TorchTitan.
# Phase-1 runs with N ranks, checkpoints at DROP_STEP, and exits.
# Phase-2 relaunches with N-1 ranks, restores checkpoint state, and skips dataloader state restore.

set -euo pipefail

DROP_STEP=${DROP_STEP:?Set DROP_STEP to the boundary step, e.g. 200}
DROP_RANK=${DROP_RANK:-1}

PHASE1_NNODES=${PHASE1_NNODES:-${NNODES:-1}}
PHASE1_NPROC_PER_NODE=${PHASE1_NPROC_PER_NODE:-${NRANK_PER_NODE:-8}}

if [[ ${PHASE1_NPROC_PER_NODE} -le 1 ]]; then
  echo "PHASE1_NPROC_PER_NODE must be > 1 to drop one rank."
  exit 1
fi

PHASE2_NNODES=${PHASE2_NNODES:-${PHASE1_NNODES}}
PHASE2_NPROC_PER_NODE=${PHASE2_NPROC_PER_NODE:-$((PHASE1_NPROC_PER_NODE - 1))}

# Keep local microbatch fixed and use a smaller effective global batch after rank drop.
# -1 means auto global batch = local_batch_size * dp_degree.
PHASE2_GLOBAL_BATCH_SIZE=${PHASE2_GLOBAL_BATCH_SIZE:--1}

CONFIG_FILE=${CONFIG_FILE:-"./torchtitan/models/llama3/train_configs/debug_model.toml"}
TRAIN_FILE=${TRAIN_FILE:-"torchtitan.train"}
MODEL_FLAVOR=${MODEL_FLAVOR:-"simple_fsdp.llama3"}

EXTRA_ARGS=${EXTRA_ARGS:-""}
PHASE2_DELETE_CHECKPOINTS_ON_COMPLETE=${PHASE2_DELETE_CHECKPOINTS_ON_COMPLETE:-true}

PHASE2_CLEANUP_ARG=""
if [[ "${PHASE2_DELETE_CHECKPOINTS_ON_COMPLETE}" == "true" ]]; then
  PHASE2_CLEANUP_ARG="--checkpoint.delete_after_training_completion"
fi

echo "[Phase-1] Running until step ${DROP_STEP} with ${PHASE1_NNODES} node(s) x ${PHASE1_NPROC_PER_NODE} rank(s)/node"
NNODES=${PHASE1_NNODES} \
NRANK_PER_NODE=${PHASE1_NPROC_PER_NODE} \
CONFIG_FILE=${CONFIG_FILE} \
TRAIN_FILE=${TRAIN_FILE} \
MODEL_FLAVOR=${MODEL_FLAVOR} \
./run_train.sh \
  --checkpoint.enable \
  --training.rank_drop_step ${DROP_STEP} \
  --training.phase1_exit_after_rank_drop_checkpoint \
  --training.rank_to_drop ${DROP_RANK} \
  ${EXTRA_ARGS}

echo "[Phase-2] Relaunching with ${PHASE2_NNODES} node(s) x ${PHASE2_NPROC_PER_NODE} rank(s)/node from step ${DROP_STEP}"
NNODES=${PHASE2_NNODES} \
NRANK_PER_NODE=${PHASE2_NPROC_PER_NODE} \
CONFIG_FILE=${CONFIG_FILE} \
TRAIN_FILE=${TRAIN_FILE} \
MODEL_FLAVOR=${MODEL_FLAVOR} \
./run_train.sh \
  --checkpoint.enable \
  --checkpoint.load_step ${DROP_STEP} \
  --checkpoint.skip_dataloader_load \
  ${PHASE2_CLEANUP_ARG} \
  --training.no_phase1_exit_after_rank_drop_checkpoint \
  --training.rank_drop_step -1 \
  --training.global_batch_size ${PHASE2_GLOBAL_BATCH_SIZE} \
  ${EXTRA_ARGS}

echo "Two-phase rank-drop run completed."
