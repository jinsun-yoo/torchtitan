# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Minimal compatibility shim for ``torchtitan.models.common.linear``.

This checkout of torchtitan predates upstream's ``models/common`` refactor
(which depends on ``torch_remat`` and a new SPMD/sharding-aware ``Module``
config framework not present here). This module reproduces just the
external API surface -- a ``Config`` dataclass with ``.build()`` and
``num_linears``-way weight packing -- needed by code that targets the
upstream API, without any of the remat/SPMD machinery.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["Linear"]


class Linear(nn.Linear):
    """``nn.Linear`` that can pack ``num_linears`` logical projections.

    When ``num_linears > 1`` the weight (and bias, if present) are stored
    with shape ``[num_linears, out_features, in_features]`` and the output
    is unflattened to ``[..., num_linears, out_features]`` -- e.g. for a
    packed gate+up projection (``num_linears=2``).
    """

    @dataclass(kw_only=True, slots=True)
    class Config:
        in_features: int
        out_features: int
        num_linears: int = 1
        bias: bool = False

        def build(self) -> "Linear":
            return Linear(self)

    def __init__(self, config: Config):
        super().__init__(
            config.in_features,
            config.num_linears * config.out_features,
            bias=config.bias,
        )
        self.out_features = config.out_features
        self.num_linears = config.num_linears
        if config.num_linears > 1:
            self.weight = nn.Parameter(
                self.weight.detach().unflatten(
                    0, (config.num_linears, config.out_features)
                ),
                requires_grad=self.weight.requires_grad,
            )
            if self.bias is not None:
                self.bias = nn.Parameter(
                    self.bias.detach().unflatten(
                        0, (config.num_linears, config.out_features)
                    ),
                    requires_grad=self.bias.requires_grad,
                )

    def _flatten_weight_and_bias(self):
        weight = self.weight.flatten(0, 1) if self.num_linears > 1 else self.weight
        bias = self.bias
        if bias is not None and self.num_linears > 1:
            bias = bias.flatten(0, 1)
        return weight, bias

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight, bias = self._flatten_weight_and_bias()
        output = F.linear(input, weight, bias)
        if self.num_linears > 1:
            output = output.unflatten(-1, (self.num_linears, self.out_features))
        return output

    def extra_repr(self) -> str:
        result = nn.Linear.extra_repr(self)
        if self.num_linears > 1:
            result += f", num_linears={self.num_linears}"
        return result
