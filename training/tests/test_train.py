#!/usr/bin/env python3
"""Hermetic test for the pure training helpers in ``training/train.py``.

Builds ``PositionRecord``s in memory (no C++ tools, no filesystem) and asserts:

  - ``flatten_records`` flattens the decoded stream structure;
  - ``records_to_tensors`` produces the right shapes/values/dtypes;
  - ``policy_cross_entropy`` / ``value_mse`` are finite scalars, the policy CE
    with a distribution target is positive, and backprop flows;
  - ``train_epoch`` performs a real optimizer step (a parameter changes).

Needs torch (developer script, not ctest).

Usage:  python tests/test_train.py
"""

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch

import game_record
import model as model_mod
import train

N_PLANES = game_record.PLANES
N_POLICY = game_record.POLICY


def make_records(n: int, seed_value: int):
    """``n`` PositionRecords with recognizable planes/policy/value."""
    records = []
    for i in range(n):
        planes = [float((i + seed_value) % 2)] * N_PLANES
        policy = [0.0] * N_POLICY
        policy[i % N_POLICY] = 1.0
        records.append(
            game_record.PositionRecord(planes=planes, policy=policy,
                                       value=float((i % 3) - 1))
        )
    return records


def test_flatten_records() -> None:
    games = [make_records(2, 1), make_records(3, 2)]
    flat = train.flatten_records(games)
    assert len(flat) == 5
    # Order is preserved (game-major, position-major).
    assert flat[0].planes[0] == games[0][0].planes[0]
    assert flat[4].planes[0] == games[1][2].planes[0]


def test_records_to_tensors() -> None:
    records = make_records(4, 3)
    planes, policy, value = train.records_to_tensors(records)
    assert planes.shape == (4, 4, 15, 15), planes.shape
    assert policy.shape == (4, 225), policy.shape
    assert value.shape == (4, 1), value.shape
    assert planes.dtype == torch.float32
    assert policy.dtype == torch.float32
    assert value.dtype == torch.float32
    assert planes[0, 0, 0, 0] == records[0].planes[0]
    assert policy[1, 1] == records[1].policy[1]
    assert value[3, 0] == records[3].value


def test_losses_finite_and_backprop() -> None:
    torch.manual_seed(0)
    logits = torch.randn(4, 225, requires_grad=True)
    target = torch.rand(4, 225)
    target = target / target.sum(1, keepdim=True)  # distribution rows

    loss = train.policy_cross_entropy(logits, target)
    assert loss.isfinite()
    assert loss.item() > 0.0
    loss.backward()
    assert logits.grad is not None
    assert logits.grad.abs().sum().item() > 0.0

    pred = torch.randn(4, 1, requires_grad=True)
    tgt = torch.randn(4, 1)
    vloss = train.value_mse(pred, tgt)
    assert vloss.isfinite()
    assert vloss.item() >= 0.0
    vloss.backward()
    assert pred.grad is not None
    assert pred.grad.abs().sum().item() > 0.0


def test_train_epoch_updates_parameters() -> None:
    torch.manual_seed(1)
    records = make_records(16, 7)
    planes, policy, value = train.records_to_tensors(records)

    model = model_mod.GomokuNet(num_blocks=1, channels=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = model.policy_fc.weight.detach().clone()

    loss = train.train_epoch(model, optimizer, planes, policy, value,
                             torch.device("cpu"), batch_size=8, value_weight=1.0)
    assert math.isfinite(loss)
    assert loss > 0.0
    assert not torch.equal(before, model.policy_fc.weight.detach()), \
        "optimizer step did not change a parameter"


def main() -> None:
    test_flatten_records()
    test_records_to_tensors()
    test_losses_finite_and_backprop()
    test_train_epoch_updates_parameters()
    print("OK: train.py pure-helper tests passed")


if __name__ == "__main__":
    main()
