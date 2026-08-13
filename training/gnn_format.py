"""Pure-Python description of the ``.gnn`` weight-file format.

Mirrors ``src/core/Weights.cpp`` exactly; the C++ loader is the single source
of truth and this module must never drift from it. Standard library only — no
torch, no numpy — so the round-trip gate and any layout tooling can run
anywhere.

Byte layout (all little-endian)::

    Header: magic[9] "GOMOKUNET"  u32 version(=1)  u32 numBlocks  u32 channels
    then, in this exact order, one raw f32 tensor after another:
      inputConvW (C,4,3,3)  inputConvB (C)
      inputBnGamma (C) inputBnBeta (C) inputBnMean (C) inputBnVar (C)
      for each of numBlocks blocks:
        conv1W (C,C,3,3) conv1B (C)
        bn1Gamma (C) bn1Beta (C) bn1Mean (C) bn1Var (C)
        conv2W (C,C,3,3) conv2B (C)
        bn2Gamma (C) bn2Beta (C) bn2Mean (C) bn2Var (C)
      policyConvW (2,C,3,3) policyConvB (2)
      policyFcW (225,450)   policyFcB (225)
      valueConvW (1,C,3,3)  valueConvB (1)
      valueFc1W (256,225)   valueFc1B (256)
      valueFc2W (1,256)     valueFc2B (1)

Tensor shapes use the symbolic dimension ``"C"`` for ``channels``. BatchNorm
tensors map to ``weight`` (gamma), ``bias`` (beta), ``running_mean``,
``running_var`` in PyTorch.
"""

from __future__ import annotations

import struct

MAGIC = b"GOMOKUNET"
VERSION = 1

# Header: magic(9 bytes) + u32 version + u32 numBlocks + u32 channels.
HEADER = struct.Struct("<9sIII")
HEADER_SIZE = HEADER.size  # 9 + 4*3 = 21

# Ordered layout before the residual blocks.
_FIXED_TENSORS = [
    ("inputConvW", ("C", 4, 3, 3)),
    ("inputConvB", ("C",)),
    ("inputBnGamma", ("C",)),
    ("inputBnBeta", ("C",)),
    ("inputBnMean", ("C",)),
    ("inputBnVar", ("C",)),
]

# One residual block (repeated numBlocks times).
_BLOCK_TENSORS = [
    ("conv1W", ("C", "C", 3, 3)),
    ("conv1B", ("C",)),
    ("bn1Gamma", ("C",)),
    ("bn1Beta", ("C",)),
    ("bn1Mean", ("C",)),
    ("bn1Var", ("C",)),
    ("conv2W", ("C", "C", 3, 3)),
    ("conv2B", ("C",)),
    ("bn2Gamma", ("C",)),
    ("bn2Beta", ("C",)),
    ("bn2Mean", ("C",)),
    ("bn2Var", ("C",)),
]

# Policy + value heads (after the blocks).
_HEAD_TENSORS = [
    ("policyConvW", (2, "C", 3, 3)),
    ("policyConvB", (2,)),
    ("policyFcW", (225, 450)),
    ("policyFcB", (225,)),
    ("valueConvW", (1, "C", 3, 3)),
    ("valueConvB", (1,)),
    ("valueFc1W", (256, 225)),
    ("valueFc1B", (256,)),
    ("valueFc2W", (1, 256)),
    ("valueFc2B", (1,)),
]


def _n_floats(shape, channels):
    """Number of floats in a shape tuple with symbolic dimension ``"C"``."""
    n = 1
    for dim in shape:
        if isinstance(dim, str):
            if dim != "C":
                raise ValueError(f"unknown symbolic dimension {dim!r}")
            n *= channels
        else:
            n *= dim
    return n


def tensor_order(num_blocks, channels):
    """Expanded ordered list of ``(name, n_floats)`` mirroring ``Weights::parse``."""
    order = [(name, _n_floats(shape, channels)) for name, shape in _FIXED_TENSORS]
    for _ in range(num_blocks):
        order.extend((name, _n_floats(shape, channels)) for name, shape in _BLOCK_TENSORS)
    order.extend((name, _n_floats(shape, channels)) for name, shape in _HEAD_TENSORS)
    return order


def total_floats(num_blocks, channels):
    """Total float32 count for a model of the given size."""
    return sum(n for _, n in tensor_order(num_blocks, channels))


def pack_f32s(values):
    """Pack an iterable of floats into little-endian float32 bytes."""
    values = list(values)
    return struct.pack(f"<{len(values)}f", *values)


def serialize(num_blocks, channels, tensor_bytes):
    """Assemble a complete ``.gnn`` buffer.

    ``tensor_bytes`` is an iterable of ``bytes`` objects, one per tensor in
    ``tensor_order()``, each already the little-endian f32 encoding.
    """
    header = HEADER.pack(MAGIC, VERSION, num_blocks, channels)
    return header + b"".join(tensor_bytes)


def tensor_offsets(num_blocks, channels):
    """Byte offset of each tensor in ``tensor_order()`` in the serialized buffer."""
    offsets = []
    off = HEADER_SIZE
    for _, n in tensor_order(num_blocks, channels):
        offsets.append(off)
        off += n * 4
    return offsets
