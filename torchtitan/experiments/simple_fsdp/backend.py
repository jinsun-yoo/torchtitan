# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
from typing import Any, Union

import torch


def get_compile_backend(backend_name: str) -> Union[str, callable]:
    # return the compile backends used in SimpleFSDP training
    # Step1: check if backend_name is inside available torch.compile backends
    # Step2: check if the backend_name has been registered as a customized backend
    available_torch_backend = torch._dynamo.list_backends(exclude_tags=())
    if backend_name in available_torch_backend:
        return backend_name

    if backend_name == "aot_eager_autobucketing":
        # Perform auto optimization in aten fx-level and execute code in aot_eager backend
        # The autobucketing logic is here: https://github.com/pytorch/pytorch/pull/163960
        from torch._dynamo.backends.common import aot_autograd as aot_autograd_backend
        from torch._inductor.fx_passes.overlap_scheduling import (
            schedule_overlap_bucketing,
        )

        torch._inductor.config.test_configs.aten_fx_overlap_preserving_bucketing = True
        torch._inductor.config.test_configs.aten_fx_overlap_insert_overlap_deps = False
        torch._inductor.config.allow_buffer_reuse = False

        def aten_autobucketing_reordering_pass(
            gm: torch.fx.GraphModule, example_inputs: Any
        ) -> torch.fx.GraphModule:
            schedule_overlap_bucketing(gm)
            gm.recompile()
            return gm

        backend = aot_autograd_backend(
            fw_compiler=aten_autobucketing_reordering_pass,
            bw_compiler=aten_autobucketing_reordering_pass,
            keep_inference_input_mutations=True,
        )
    elif backend_name == "aot_eager_nooverlap":
        # aot_eager with every wait_tensor moved directly after its collective, so the
        # compute stream blocks on each collective as soon as it is issued: no
        # communication/computation overlap.
        from torch._dynamo.backends.common import aot_autograd as aot_autograd_backend

        def no_overlap_pass(
            gm: torch.fx.GraphModule, example_inputs: Any
        ) -> torch.fx.GraphModule:
            _sink_waits_to_collectives(gm)
            gm.recompile()
            return gm

        backend = aot_autograd_backend(
            fw_compiler=no_overlap_pass,
            bw_compiler=no_overlap_pass,
            keep_inference_input_mutations=True,
        )
    elif backend_name == "aot_eager_layerbucket_nooverlap":
        # aot_eager with FSDP all-gathers/reduce-scatters bucketed per top-level module
        # (tok_embeddings, layers.<i>, norm, output), then every wait_tensor moved directly
        # after its collective: one AG/RS per layer and no communication/computation overlap.
        from torch._dynamo.backends.common import aot_autograd as aot_autograd_backend

        def layerbucket_no_overlap_pass(
            gm: torch.fx.GraphModule, example_inputs: Any
        ) -> torch.fx.GraphModule:
            _bucket_per_module(gm)
            _sink_waits_to_collectives(gm)
            gm.recompile()
            return gm

        backend = aot_autograd_backend(
            fw_compiler=layerbucket_no_overlap_pass,
            bw_compiler=layerbucket_no_overlap_pass,
            keep_inference_input_mutations=True,
        )
    else:
        raise AssertionError(f"Unsupported customized backend: {backend_name}")

    return backend


_STACK_NOISE = {"self", "__dict__", "_modules", "_checkpoint_wrapped_module", "_orig_mod", "getattr"}


def _bucket_per_module(gm: torch.fx.GraphModule) -> None:
    """Merge FSDP all-gathers/reduce-scatters of each top-level module (layers.<i>
    counts as one module) into a single collective, using the bucketing step of
    torch's ManualOverlapScheduler without its overlap reordering."""
    from collections import Counter

    from torch._inductor.fx_passes.bucketing import (
        is_all_gather_into_tensor,
        is_reduce_scatter_tensor,
    )
    from torch._inductor.fx_passes.graph_view import _get_module_stack
    from torch._inductor.fx_passes.overlap_manual_scheduling import (
        ManualOverlapScheduler,
    )

    from torchtitan.tools.logging import logger

    def top_module(node: torch.fx.Node) -> str | None:
        # Skip SimpleFSDP's parametrization closure entries ("dict(type(...)...fget...").
        # Parse identifier tokens rather than _clean_stack_name: with activation checkpointing the
        # stack reads "L['self'].layers.0.__dict__['_modules']['_checkpoint_wrapped_module'].w",
        # which _clean_stack_name reduces to "_modules._checkpoint_wrapped_module".
        for s, _ in reversed(_get_module_stack(node)):
            if s.startswith("dict("):
                continue
            toks = [t for t in re.findall(r"\w+", re.sub(r"^L\['self'\]", "", s))
                    if t not in _STACK_NOISE]
            for i, t in enumerate(toks[:-1]):
                if t == "layers" and toks[i + 1].isdigit():
                    return f"layers.{toks[i + 1]}"
            if toks:
                return toks[0]
        return None

    def count() -> Counter:
        return Counter(
            "AG" if is_all_gather_into_tensor(n) else "RS"
            for n in gm.graph.nodes
            if is_all_gather_into_tensor(n) or is_reduce_scatter_tensor(n)
        )

    plans: list[str] = []
    for n in gm.graph.nodes:
        if is_all_gather_into_tensor(n) or is_reduce_scatter_tensor(n):
            m = top_module(n)
            if m is not None and m not in plans:
                plans.append(m)
    before = count()
    if len(plans) < 2:
        logger.warning(
            f"aot_eager_layerbucket_nooverlap: only {plans} module buckets found; skipping bucketing"
        )
        plans = []
    if plans:

        class _BucketOnly(ManualOverlapScheduler):
            def run(self) -> torch.fx.GraphModule:
                self._manual_bucket_collectives()
                return self.gm

        _BucketOnly(gm, plans, insert_overlap_deps=False).run()
        gm.graph.lint()
    logger.info(
        f"aot_eager_layerbucket_nooverlap: {len(plans)} module buckets {plans[:3]}..{plans[-2:]}, "
        f"collectives {dict(before)} -> {dict(count())}"
    )


def _sink_waits_to_collectives(gm: torch.fx.GraphModule) -> None:
    import operator

    from torchtitan.tools.logging import logger

    c10d = torch.ops._c10d_functional
    wait = c10d.wait_tensor.default
    moved = total = 0
    for node in list(gm.graph.nodes):
        if node.op != "call_function" or node.target is not wait:
            continue
        total += 1
        src = node.args[0]
        # coalesced collectives: wait_tensor(getitem(coll, i)) -> place after the getitem
        anchor = src
        if src.op == "call_function" and src.target is operator.getitem:
            src = src.args[0]
        if not (
            src.op == "call_function"
            and getattr(src.target, "namespace", None) == "_c10d_functional"
        ):
            continue
        if anchor.next is not node:
            anchor.append(node)
            moved += 1
    gm.graph.lint()
    logger.info(f"aot_eager_nooverlap: {total} wait_tensor nodes, {moved} moved next to their collective")
