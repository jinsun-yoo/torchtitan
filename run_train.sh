#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex
# Check if running in MPI context
if [ -z "$OMPI_COMM_WORLD_SIZE" ] && [ -z "$PMI_SIZE" ] && [ -z "$MPI_LOCALNRANKS" ]; then
  echo "Error: Not running in MPI context. Please run with mpirun/mpiexec."
  exit 1
fi
# mpirun -np 8 -N 1 CONFIG_FILE="" ./run_train.sh
# use envs as local overwrites for convenience
# e.g.
# LOG_RANK=0,1 NGPU=4 ./run_train.sh
NNODES=${NNODES:-"4"}
NRANK_PER_NODE=${NRANK_PER_NODE:-"1"}
export LOG_RANK=${LOG_RANK:-0}
CONFIG_FILE=${CONFIG_FILE:-"./torchtitan/models/llama3/train_configs/debug_model.toml"}
TRAIN_FILE=${TRAIN_FILE:-"torchtitan.train"}
MODEL_NAME=${MODEL_NAME:-"simple_fsdp.llama3"}

RDZV_MASTER_HOSTNAME=${RDZV_MASTER_HOSTNAME:-"g100n052"}
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"$RDZV_MASTER_HOSTNAME:29510"}

PYTORCH_ALLOC_CONF="expandable_segments:True" \
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE} \
torchrun --nnodes=${NNODES} --nproc-per-node=${NRANK_PER_NODE} \
  --rdzv-id=182745 --rdzv-backend=c10d --rdzv-endpoint=$RDZV_MASTER_HOSTNAME:29500 \
  -m ${TRAIN_FILE} --job.config_file ${CONFIG_FILE} --model.name ${MODEL_NAME} "$@"
#   --local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
