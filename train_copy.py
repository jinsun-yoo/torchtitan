# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import os

import torch
from torch.distributed.elastic.multiprocessing.errors import record
from functorch.compile import make_boxed_func
from torch._dynamo.backends.common import aot_autograd

from torchtitan import utils
from torchtitan.checkpoint import TrainState
from torchtitan.config_manager import JobConfig
from torchtitan.datasets import build_hf_data_loader, build_tokenizer
from torchtitan.logging import  logger
from torchtitan.models import model_name_to_cls, model_name_to_tokenizer, models_config
from torchtitan.config_manager import JobConfig, TORCH_DTYPE_MAP
from torchtitan.profiling import maybe_enable_memory_snapshot, maybe_enable_profiling
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy


def get_train_context(enable_loss_parallel: bool, enable_compiled_autograd: bool):
    @contextlib.contextmanager
    def context():
        with contextlib.ExitStack() as stack:
            if enable_loss_parallel:
                stack.enter_context(torch.distributed.tensor.parallel.loss_parallel())
            if enable_compiled_autograd:
                stack.enter_context(
                    torch._dynamo.utils.maybe_enable_compiled_autograd(True)
                )
            yield

    return context


# Enable debug tracing on failure: https://pytorch.org/docs/stable/elastic/errors.html
@record
def main(job_config: JobConfig):

    # init distributed
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    torch.cuda.set_device(device)
    utils.init_distributed(job_config)
    # initialize GPU memory monitor and get peak flops for MFU calculation

    # build meshes
    world_mesh = init_device_mesh("cuda", [world_size])
    dp_degree, dp_rank = 8, int(os.environ['RANK'])
    model_name = job_config.model.name

    # build tokenizer
    tokenizer_type = model_name_to_tokenizer[model_name]
    tokenizer = build_tokenizer(tokenizer_type, job_config.model.tokenizer_path)

    # build dataloader
    data_loader = build_hf_data_loader(
        job_config.training.dataset,
        job_config.training.dataset_path,
        tokenizer,
        job_config.training.batch_size,
        job_config.training.seq_len,
        dp_degree,
        dp_rank,
    )

    # build model (using meta init)
    model_cls = model_name_to_cls[model_name]
    model_config = models_config[model_name][job_config.model.flavor]

    model_config.norm_type = job_config.model.norm_type
    model_config.vocab_size = tokenizer.n_words
    model_config.max_seq_len = job_config.training.seq_len

    logger.info(f"Building {model_name} {job_config.model.flavor} with {model_config}")
    with torch.device("meta"):
        model = model_cls.from_model_args(model_config)

    # apply FSDP parallel
    param_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_param]
    reduce_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_reduce]
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    fsdp_config = {"mesh": world_mesh, "mp_policy": mp_policy}

    for layer_id, transformer_block in model.layers.named_children():
        transformer_block = torch.compile(transformer_block, fullgraph=True)
        model.layers.register_module(layer_id, transformer_block)

    # move sharded model to CPU/GPU and initialize weights via DTensor
    model.to_empty(device="cuda")
    model.init_weights()
    model.train()

    def custom_backend(gm: torch.fx.GraphModule, list_sample):
        if os.environ['LOCAL_RANK'] != '0':
            return make_boxed_func(gm.forward)

        for node in gm.graph.nodes:
            print(node.name)
            print(node.target)
        return make_boxed_func(gm.forward) 

    model = torch.compile(model, backend=aot_autograd(fw_compiler=custom_backend))

    train_state = TrainState()
    data_iterator = iter(data_loader)
    loss_parallel_enabled = False
    train_context = get_train_context(
        loss_parallel_enabled,
        job_config.experimental.enable_compiled_autograd,
    )

    with maybe_enable_profiling(
        job_config, global_step=train_state.step
    ) as torch_profiler, maybe_enable_memory_snapshot(
        job_config, global_step=train_state.step
    ) as memory_profiler:
        # get batch
        batch = next(data_iterator)
        input_ids, labels = batch

        input_ids = input_ids.cuda()
        labels = labels.cuda()

        # Forward pass
        with train_context():
            model(input_ids)



if __name__ == "__main__":
    config = JobConfig()
    config.parse_args()
    main(config)
    torch.distributed.destroy_process_group()
