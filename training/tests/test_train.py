#!/usr/bin/env python3
"""Hermetic test for the pure training helpers in ``training/train.py``.

Builds ``PositionRecord``s in memory (no C++ tools) and asserts:

  - ``flatten_records`` flattens the decoded stream structure;
  - ``records_to_tensors`` produces the right shapes/values/dtypes;
  - ``policy_cross_entropy`` / ``value_mse`` are finite scalars, the policy CE
    with a distribution target is positive, and backprop flows;
  - ``train_epoch`` performs a real optimizer step (a parameter changes);
  - ``balanced_offsets`` and ``merge_rec_files`` (the ``--jobs`` helpers) behave
    correctly (the merge test uses a stdlib ``tempfile`` dir, no C++ tools).

Needs torch (developer script, not ctest).

Usage:  python tests/test_train.py
"""

import math
import pathlib
import sys
import tempfile

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


def test_balanced_offsets() -> None:
    """``balanced_offsets`` produces balanced chunk start offsets."""
    # Exact division.
    assert train.balanced_offsets(8, 4) == [0, 2, 4, 6]
    # Remainder goes to the first chunks (sizes differ by at most one).
    assert train.balanced_offsets(7, 3) == [0, 3, 5]
    # jobs > total clamps to total single-game chunks.
    assert train.balanced_offsets(3, 10) == [0, 1, 2]
    # A single job starts at zero.
    assert train.balanced_offsets(5, 1) == [0]


def test_merge_rec_files() -> None:
    """``merge_rec_files`` concatenates streams at the byte level."""
    games_a = [make_records(2, 1), make_records(3, 2)]
    games_b = [make_records(4, 3)]
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        a = td / "a.rec"
        b = td / "b.rec"
        out = td / "merged.rec"
        a.write_bytes(game_record.encode_stream(games_a))
        b.write_bytes(game_record.encode_stream(games_b))
        train.merge_rec_files([a, b], out)
        merged = game_record.decode_stream(out.read_bytes())
        assert len(merged) == 3
        assert len(merged[0]) == 2
        assert len(merged[1]) == 3
        assert len(merged[2]) == 4


def test_merge_rec_files_rejects_bad_streams() -> None:
    """``merge_rec_files`` rejects a bad magic/version and leaves no partial out."""
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        good = td / "good.rec"
        bad = td / "bad.rec"
        out = td / "merged.rec"
        good.write_bytes(game_record.encode_stream([make_records(1, 1)]))
        bad.write_bytes(b"GOMOKUREC" + bytes(4 + 1024))  # bad version bytes
        try:
            train.merge_rec_files([good, bad], out)
        except ValueError:
            pass
        else:
            raise AssertionError("merge_rec_files did not reject a bad stream")
        assert not out.exists(), "merge_rec_files wrote a partial output on failure"


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
    test_balanced_offsets()
    test_merge_rec_files()
    test_merge_rec_files_rejects_bad_streams()
    print("OK: train.py pure-helper tests passed")


if __name__ == "__main__":
    main()
