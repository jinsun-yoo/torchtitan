# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import os
import pickle
import shutil
import time

import torch

from torchtitan.config import Profiling as ProfilingConfig
from torchtitan.tools.logging import logger
from torch.profiler import ExecutionTraceObserver, profile

# how much memory allocation/free ops to record in memory snapshots
MEMORY_SNAPSHOT_MAX_ENTRIES = 100000


def _enable_cupti_cuda_event_on_warmup(torch_profiler) -> None:
    """Opt-in (TT_PROFILE_CUPTI_CUDA_EVENT=1): also enable CUPTI_ACTIVITY_KIND_CUDA_EVENT.

    Kineto (torch 2.13 / CUDA 13) enables only CUPTI_ACTIVITY_KIND_SYNCHRONIZATION for
    enable_cuda_sync_events, so "Stream Wait Event" records carry wait_on_stream /
    wait_on_cuda_event_record_corr_id = -1 and no "cuda_event" records are exported. Kineto already
    handles CUDA_EVENT records when they arrive, so we enable the kind ourselves right after Kineto
    has enabled its activities (the transition into WARMUP/RECORD).
    """
    import ctypes

    state = {"done": False}
    orig_step = torch_profiler.step
    live = (
        torch.profiler.ProfilerAction.WARMUP,
        torch.profiler.ProfilerAction.RECORD,
        torch.profiler.ProfilerAction.RECORD_AND_SAVE,
    )

    def step(*args, **kwargs):
        ret = orig_step(*args, **kwargs)
        if not state["done"] and torch_profiler.current_action in live:
            state["done"] = True
            try:
                rc = ctypes.CDLL("libcupti.so.13").cuptiActivityEnable(ctypes.c_int(36))
                logger.info(f"cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CUDA_EVENT) -> {rc}")
            except OSError as e:
                logger.warning(f"could not enable CUPTI CUDA_EVENT activity: {e}")
        return ret

    torch_profiler.step = step


@contextlib.contextmanager
def maybe_enable_profiling(
    profiling_config: ProfilingConfig,
    *,
    global_step: int = 0,
    base_folder: str = "",
    leaf_folder: str = "",
):
    # get user defined profiler settings
    enable_profiling = profiling_config.enable_profiling

    if enable_profiling:
        trace_dir = os.path.join(base_folder, profiling_config.save_traces_folder)
        profile_freq, warmup, active = (
            profiling_config.profile_freq,
            profiling_config.profiler_warmup,
            profiling_config.profiler_active,
        )

        rank = torch.distributed.get_rank()

        if not os.path.exists(trace_dir):
            os.makedirs(trace_dir, exist_ok=True)
        # Opt-out switches (default = previous behaviour, all on):
        #   TT_PROFILE_ET=0            no ExecutionTraceObserver (no pytorch_et_rank_*.json)
        #   TT_PROFILE_RECORD_SHAPES=0 record_shapes=False (Kineto "Input Dims/type/Strides"; Chakra takes shapes from ET)
        #   TT_PROFILE_CUDA_SYNC=0     no enable_cuda_sync_events (no cudaStreamWaitEvent/cudaEventRecord sync records)
        use_et = os.environ.get("TT_PROFILE_ET", "1") != "0"
        record_shapes = os.environ.get("TT_PROFILE_RECORD_SHAPES", "1") != "0"
        cuda_sync = os.environ.get("TT_PROFILE_CUDA_SYNC", "1") != "0"
        et = None
        if use_et:
            et = ExecutionTraceObserver()
            et.register_callback(f"{trace_dir}/pytorch_et_rank_{rank}.json")
        logger.info(
            f"Profiler options: execution_trace_observer={use_et} record_shapes={record_shapes} "
            f"enable_cuda_sync_events={cuda_sync}"
        )

        # Opt-in (TT_PROFILE_PER_STEP_TRACES=1): keep every dumped cycle as
        # kineto_rank{rank}_step{N}.json; kineto_rank{rank}_trace.json is a hard link to the latest one.
        per_step = os.environ.get("TT_PROFILE_PER_STEP_TRACES", "0") == "1"

        def trace_handler(prof):
            logger.info(f"Dumping profiler traces at step {prof.step_num}")
            begin = time.monotonic()

            output_file = os.path.join(trace_dir, f"kineto_rank{rank}_trace.json")
            if per_step:
                step_file = os.path.join(
                    trace_dir, f"kineto_rank{rank}_step{prof.step_num}.json"
                )
                if os.path.exists(step_file):
                    # Profiler exit after the last step re-fires the handler with the same step_num and a
                    # near-empty cycle; keep it separately instead of overwriting the last real step.
                    prof.export_chrome_trace(step_file[: -len(".json")] + "_exit.json")
                else:
                    prof.export_chrome_trace(step_file)
                    if os.path.lexists(output_file):
                        os.remove(output_file)
                    try:
                        os.link(step_file, output_file)
                    except OSError:
                        shutil.copyfile(step_file, output_file)
            else:
                prof.export_chrome_trace(output_file)
            logger.info(
                f"Finished dumping profiler traces in {time.monotonic() - begin:.2f} seconds"
            )

        logger.info(f"Profiling active. Traces will be saved at {trace_dir}")

        if not os.path.exists(trace_dir):
            os.makedirs(trace_dir, exist_ok=True)

        wait = profile_freq - (active + warmup)
        assert (
            wait >= 0
        ), "profile_freq must be greater than or equal to warmup + active"
        print(f"Profiler will wait for {wait} steps, warmup for {warmup} steps, then be active for {active} steps, global steps {global_step}.")
        gpu_device_profiled = None
        if torch.cuda.is_available():
            gpu_device_profiled = torch.profiler.ProfilerActivity.CUDA
        elif torch.xpu.is_available():
            gpu_device_profiled = torch.profiler.ProfilerActivity.XPU
        # Opt-in TT_PROFILE_ACTIVITIES=cpu|cuda|cpu,cuda (default cpu,cuda = previous behaviour).
        #   cuda: no CPU activity -> Kineto installs no RecordFunction callbacks (no cpu_op / ProfilerStep#N);
        #         CUPTI still records kernels, memcpy/memset and cuda_runtime/driver calls with correlation ids.
        #   cpu:  no device activity -> CUPTI activities are never enabled (RecordFunction cpu_op events only).
        act_names = {
            a.strip().lower()
            for a in os.environ.get("TT_PROFILE_ACTIVITIES", "cpu,cuda").split(",")
            if a.strip()
        }
        if not act_names or not act_names <= {"cpu", "cuda"}:
            raise ValueError(f"TT_PROFILE_ACTIVITIES must be cpu, cuda or cpu,cuda; got {act_names}")
        activities = []
        if "cpu" in act_names:
            activities.append(torch.profiler.ProfilerActivity.CPU)
        if "cuda" in act_names and gpu_device_profiled is not None:
            activities.append(gpu_device_profiled)
        logger.info(f"Profiler activities: {activities}")
        with torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active),
            on_trace_ready=trace_handler,
            execution_trace_observer=et,
            record_shapes=record_shapes,
            # Capture CUDA_SYNC activities (cudaStreamWaitEvent/cudaEventRecord annotated with the
            # stream and paired event-record correlation they act on), so downstream tools (e.g. HTA's
            # critical path analysis) can resolve cross-stream dependencies deterministically instead
            # of guessing from dispatch order.
            experimental_config=torch._C._profiler._ExperimentalConfig(
                enable_cuda_sync_events=cuda_sync
            ),
        ) as torch_profiler:
            torch_profiler.step_num = global_step
            if os.environ.get("TT_PROFILE_CUPTI_CUDA_EVENT", "0") == "1":
                _enable_cupti_cuda_event_on_warmup(torch_profiler)
            yield torch_profiler
    else:
        torch_profiler = contextlib.nullcontext()
        yield None


@contextlib.contextmanager
def maybe_enable_memory_snapshot(
    profiling_config: ProfilingConfig,
    *,
    global_step: int = 0,
    base_folder: str = "",
    leaf_folder: str = "",
):
    enable_snapshot = profiling_config.enable_memory_snapshot
    if enable_snapshot:
        snapshot_dir = os.path.join(
            base_folder, profiling_config.save_memory_snapshot_folder
        )
        if not os.path.exists(snapshot_dir):
            os.makedirs(snapshot_dir, exist_ok=True)
        rank = torch.distributed.get_rank()

        class MemoryProfiler:
            def __init__(self, step_num: int, freq: int):
                torch.cuda.memory._record_memory_history(
                    max_entries=MEMORY_SNAPSHOT_MAX_ENTRIES
                )
                # when resume training, we start from the last step
                self.step_num = step_num
                self.freq = freq

            def step(self, exit_ctx: bool = False):
                self.step_num += 1
                if not exit_ctx and self.step_num % self.freq != 0:
                    return
                if not exit_ctx:
                    curr_step = self.step_num
                    dir_name = f"iteration_{curr_step}"
                else:
                    # dump as iteration_0_exit if OOM at iter 1
                    curr_step = self.step_num - 1
                    dir_name = f"iteration_{curr_step}_exit"
                curr_snapshot_dir = os.path.join(snapshot_dir, dir_name, leaf_folder)
                if not os.path.exists(curr_snapshot_dir):
                    os.makedirs(curr_snapshot_dir, exist_ok=True)
                logger.info(f"Dumping memory snapshot at step {curr_step}")
                begin = time.monotonic()
                output_file = os.path.join(
                    curr_snapshot_dir, f"rank{rank}_memory_snapshot.pickle"
                )
                with open(output_file, "wb") as output:
                    pickle.dump(torch.cuda.memory._snapshot(), output)
                logger.info(
                    f"Finished dumping memory snapshot in {time.monotonic() - begin:.2f} seconds"
                )

        logger.info(f"Memory profiler active. Snapshot will be saved at {snapshot_dir}")
        profiler = MemoryProfiler(global_step, profiling_config.profile_freq)
        try:
            yield profiler
        except torch.OutOfMemoryError as e:
            profiler.step(exit_ctx=True)
    else:
        yield None
