"""Export a ``GomokuNet`` to the ``.gnn`` byte format.

The tensor order is pinned by ``src/core/Weights.cpp`` (the single source of
truth) and mirrored in Python by ``gnn_format``. Tensors are read by **explicit
attribute access** (never by ``state_dict()`` key strings) so a rename in
``model.py`` cannot silently reorder the file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

import gnn_format


def ordered_tensors(model):
    """Yield every tensor in the exact ``Weights::parse`` order.

    ``model`` must be a ``GomokuNet`` (``model.py``) with submodules
    ``input_conv/input_bn``, ``blocks`` (ModuleList of ``ResidualBlock`` with
    ``conv1/bn1/conv2/bn2``), ``policy_conv/policy_fc`` and
    ``value_conv/value_fc1/value_fc2``.
    """
    yield model.input_conv.weight
    yield model.input_conv.bias
    yield model.input_bn.weight
    yield model.input_bn.bias
    yield model.input_bn.running_mean
    yield model.input_bn.running_var
    for block in model.blocks:
        yield block.conv1.weight
        yield block.conv1.bias
        yield block.bn1.weight
        yield block.bn1.bias
        yield block.bn1.running_mean
        yield block.bn1.running_var
        yield block.conv2.weight
        yield block.conv2.bias
        yield block.bn2.weight
        yield block.bn2.bias
        yield block.bn2.running_mean
        yield block.bn2.running_var
    yield model.policy_conv.weight
    yield model.policy_conv.bias
    yield model.policy_fc.weight
    yield model.policy_fc.bias
    yield model.value_conv.weight
    yield model.value_conv.bias
    yield model.value_fc1.weight
    yield model.value_fc1.bias
    yield model.value_fc2.weight
    yield model.value_fc2.bias


def export_gnn(model) -> bytes:
    """Serialize a ``GomokuNet`` to ``.gnn`` bytes (little-endian f32)."""
    order = gnn_format.tensor_order(model.num_blocks, model.channels)
    tensors = list(ordered_tensors(model))
    if len(tensors) != len(order):
        raise AssertionError(
            f"model has {len(tensors)} tensors but layout has {len(order)}"
        )
    packed = []
    for tensor, (name, n) in zip(tensors, order):
        flat = tensor.detach().cpu().to(torch.float32).contiguous().view(-1)
        if flat.numel() != n:
            raise AssertionError(
                f"{name}: model has {flat.numel()} floats, layout expects {n}"
            )
        packed.append(flat.numpy().astype("<f4").tobytes())
    return gnn_format.serialize(model.num_blocks, model.channels, packed)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="output .gnn path")
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0, help="torch RNG seed (0 = unseeded)")
    args = parser.parse_args(argv)

    # Weights.cpp only accepts numBlocks/channels in [1, 1024]; reject anything
    # else up front so a user cannot export a .gnn the C++ loader will refuse.
    if not 1 <= args.num_blocks <= 1024:
        parser.error("--num-blocks must be in [1, 1024]")
    if not 1 <= args.channels <= 1024:
        parser.error("--channels must be in [1, 1024]")

    import model as model_mod

    if args.seed:
        torch.manual_seed(args.seed)
    net = model_mod.GomokuNet(num_blocks=args.num_blocks, channels=args.channels)
    net.eval()
    Path(args.out).write_bytes(export_gnn(net))
    print(
        f"wrote {args.out} ({net.num_blocks} blocks, {net.channels} channels, "
        f"{gnn_format.total_floats(net.num_blocks, net.channels) * 4} bytes of tensors)"
    )


if __name__ == "__main__":
    main()
