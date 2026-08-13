"""PyTorch model matching the C++ Stage-2 network (``src/core/NeuralNet.cpp``).

Architecture (channels-first, mirroring ``NeuralNet::evaluate``):

- Input: ``(B, 4, 15, 15)`` — four binary planes (to-move stones, opponent
  stones, constant 1, last-move marker) as produced by ``BoardEncoder::encode``.
- Input projection: ``conv3x3(4 -> C, pad 1) -> BN -> ReLU``.
- Body: ``num_blocks`` residual blocks, each
  ``conv3x3(C) -> BN -> ReLU -> conv3x3(C) -> BN -> add-input -> ReLU``.
- Policy head: ``conv3x3(C -> 2) -> flatten(2*225, channel-major) ->
  linear(450 -> 225)`` raw logits (no softmax, no masking).
- Value head: ``conv3x3(C -> 1) -> flatten(225) -> linear(225 -> 256) -> ReLU ->
  linear(256 -> 1) -> tanh``.

BatchNorm is eval-mode running stats: ``track_running_stats=True``,
``eps=1e-5`` — exactly what the C++ side stores in the ``.gnn``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """One residual block: ReLU(x + BN2(conv2(ReLU(BN1(conv1(x))))))."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(channels, eps=1e-5, track_running_stats=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(channels, eps=1e-5, track_running_stats=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.bn1(self.conv1(x))
        h = F.relu(h)
        h = self.bn2(self.conv2(h))
        return F.relu(x + h)


class GomokuNet(nn.Module):
    """Policy/value network for gomoku, layout-compatible with the C++ engine.

    Submodule names and tensor order are load-bearing: ``export_gnn.py`` reads
    them by explicit attribute access in the exact ``Weights::parse`` order.
    """

    BOARD = 15

    def __init__(self, num_blocks: int = 4, channels: int = 16) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.channels = channels

        self.input_conv = nn.Conv2d(4, channels, 3, padding=1, bias=True)
        self.input_bn = nn.BatchNorm2d(channels, eps=1e-5, track_running_stats=True)

        self.blocks = nn.ModuleList([ResidualBlock(channels) for _ in range(num_blocks)])

        self.policy_conv = nn.Conv2d(channels, 2, 3, padding=1, bias=True)
        self.policy_fc = nn.Linear(2 * self.BOARD * self.BOARD, self.BOARD * self.BOARD)

        self.value_conv = nn.Conv2d(channels, 1, 3, padding=1, bias=True)
        self.value_fc1 = nn.Linear(self.BOARD * self.BOARD, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(policy_logits (B, 225), value (B, 1))``.

        ``policy_logits`` are raw logits (no softmax); ``value`` is in [-1, 1].
        """
        x = F.relu(self.input_bn(self.input_conv(x)))
        for block in self.blocks:
            x = block(x)

        p = self.policy_conv(x)  # (B, 2, 15, 15)
        p = p.flatten(1)  # channel-major flatten: 2*225
        policy_logits = self.policy_fc(p)  # (B, 225)

        v = self.value_conv(x)  # (B, 1, 15, 15)
        v = v.flatten(1)  # 225
        v = F.relu(self.value_fc1(v))  # (B, 256)
        v = torch.tanh(self.value_fc2(v))  # (B, 1)

        return policy_logits, v
