# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Minimal compatibility shim for ``torchtitan.models.common.feed_forward``.

See ``torchtitan.models.common.linear`` for why this shim exists. It
reproduces the upstream external API (``Config``/``.build()``, a packed
``w13`` gate+up projection with ``num_linears=2``, SwiGLU activation) without
the ``torch_remat``/SPMD machinery the real upstream module depends on.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtitan.models.common.linear import Linear

__all__ = ["FeedForward", "compute_ffn_hidden_dim"]


def compute_ffn_hidden_dim(
    dim: int,
    *,
    multiple_of: int = 1,
    ffn_dim_multiplier: float | None = None,
) -> int:
    """Compute the SwiGLU hidden dimension for Llama3/4-style models.

    This applies the 2/3 scaling, optional multiplier, and rounds up to multiple_of.
    """
    hidden_dim = int(2 * 4 * dim / 3)
    if ffn_dim_multiplier is not None:
        hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    return multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)


class FeedForward(nn.Module):
    """SwiGLU feed-forward with one physical gate-and-up projection.

    ``w13`` stores the gate and up projections as ``[2, F, D]``. Config takes
    the **final** hidden_dim (no internal 2/3 scaling). Use
    ``compute_ffn_hidden_dim()`` for Llama3/4-style dim computation.
    """

    @dataclass(kw_only=True, slots=True)
    class Config:
        w13: Linear.Config
        w2: Linear.Config

        def build(self) -> "FeedForward":
            return FeedForward(self)

    def __init__(self, config: Config):
        super().__init__()
        if config.w13.num_linears != 2:
            raise ValueError(
                "FeedForward w13 requires num_linears=2 for gate/up projections"
            )
        self.w13 = config.w13.build()
        self.w2 = config.w2.build()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_TF, up_TF = self.w13(x).unbind(-2)
        return self.w2(F.silu(gate_TF) * up_TF)

    def init_weights(self, init_std: float) -> None:
        nn.init.trunc_normal_(self.w13.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.w2.weight, mean=0.0, std=init_std)
