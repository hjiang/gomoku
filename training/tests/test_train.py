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


def test_dihedral_transform_anchors() -> None:
    """k=0 identity; k=1 pure rot90 maps (3,5)->(9,3); k=4 pure flip maps (3,5)->(3,9)."""
    marker = (3, 5)
    planes = torch.zeros(1, 4, 15, 15)
    planes[0, 0, marker[0], marker[1]] = 1.0
    policy = torch.zeros(1, 225)
    policy[0, marker[0] * 15 + marker[1]] = 1.0

    def cell(t: torch.Tensor) -> tuple:
        flat = int(torch.argmax(t[0, 0].reshape(-1)).item())
        return (flat // 15, flat % 15)

    # k=0: identity.
    p0, pol0 = train.dihedral_transform(planes.clone(), policy.clone(), 0)
    assert cell(p0) == marker
    assert int(torch.argmax(pol0).item()) == marker[0] * 15 + marker[1]
    # k=1: pure rot90 (counterclockwise) maps (r,c) -> (14-c, r).
    p1, pol1 = train.dihedral_transform(planes.clone(), policy.clone(), 1)
    assert cell(p1) == (9, 3)
    assert int(torch.argmax(pol1).item()) == 9 * 15 + 3
    # k=4: pure horizontal flip maps (r,c) -> (r, 14-c).
    p4, pol4 = train.dihedral_transform(planes.clone(), policy.clone(), 4)
    assert cell(p4) == (3, 9)
    assert int(torch.argmax(pol4).item()) == 3 * 15 + 9


def test_dihedral_transform_consistency() -> None:
    """For all k: plane marker cell == policy argmax cell (same spatial map)."""
    rng = torch.Generator().manual_seed(123)
    for k in range(8):
        planes = torch.zeros(2, 4, 15, 15)
        policy = torch.zeros(2, 225)
        for b in range(2):
            cell = int(torch.randint(0, 225, (1,), generator=rng).item())
            r, c = divmod(cell, 15)
            planes[b, 0, r, c] = 1.0
            policy[b, cell] = 1.0
        tp, tpol = train.dihedral_transform(planes, policy, k)
        for b in range(2):
            plane_cell = int(torch.argmax(tp[b, 0].reshape(-1)).item())
            pol_cell = int(torch.argmax(tpol[b]).item())
            assert plane_cell == pol_cell, (k, b, plane_cell, pol_cell)


def test_dihedral_transform_permutation() -> None:
    """For all k: policy row sums and marker-plane ones counts are preserved."""
    rng = torch.Generator().manual_seed(999)
    planes = torch.zeros(2, 4, 15, 15)
    policy = torch.rand(2, 225)
    policy = policy / policy.sum(1, keepdim=True)
    for b in range(2):
        for ch in range(4):
            n_ones = 3 + ch
            cells = torch.randperm(225, generator=rng)[:n_ones]
            planes[b, ch].view(-1).scatter_(0, cells, 1.0)
    for k in range(8):
        tp, tpol = train.dihedral_transform(planes, policy, k)
        assert torch.equal(tp.sum(dim=(2, 3)), planes.sum(dim=(2, 3))), k
        assert torch.allclose(tpol.sum(1), policy.sum(1)), k
        assert torch.equal(tp, tp.round()), k


def test_split_records() -> None:
    """Deterministic, disjoint, covers all indices; val_frac=0 -> empty val."""
    records = make_records(100, 5)

    tr0, va0 = train.split_records(records, 0.0, 42)
    assert len(va0) == 0
    assert len(tr0) == 100

    tr1, va1 = train.split_records(records, 0.25, 7)
    tr2, va2 = train.split_records(records, 0.25, 7)
    assert len(va1) == 25
    assert [id(r) for r in tr1] == [id(r) for r in tr2]
    assert [id(r) for r in va1] == [id(r) for r in va2]

    ids_tr = {id(r) for r in tr1}
    ids_va = {id(r) for r in va1}
    assert ids_tr.isdisjoint(ids_va)
    assert ids_tr | ids_va == {id(r) for r in records}

    tr3, va3 = train.split_records(records, 0.25, 8)
    assert [id(r) for r in va1] != [id(r) for r in va3]
    try:
        train.split_records(records, 1.0, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("val_frac >= 1 should be rejected")


def test_train_epoch_augmented_steps_parameters() -> None:
    """With per-sample dihedral augmentation active, train_epoch still steps params."""
    torch.manual_seed(11)
    records = make_records(16, 9)
    planes, policy, value = train.records_to_tensors(records)

    model = model_mod.GomokuNet(num_blocks=1, channels=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = model.policy_fc.weight.detach().clone()

    loss = train.train_epoch(model, optimizer, planes, policy, value,
                             torch.device("cpu"), batch_size=8, value_weight=1.0)
    assert math.isfinite(loss)
    assert not torch.equal(before, model.policy_fc.weight.detach()), \
        "augmented optimizer step did not change a parameter"


def test_train_on_records_val() -> None:
    """train_on_records returns per-epoch metrics; val keys only when val_frac>0."""
    torch.manual_seed(3)
    records = make_records(24, 2)
    model = model_mod.GomokuNet(num_blocks=1, channels=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    no_val = train.train_on_records(
        model, records, device=torch.device("cpu"), epochs=1, batch_size=8,
        optimizer=optimizer, value_weight=1.0, label="no-val", val_frac=0.0,
        val_seed=0,
    )
    with_val = train.train_on_records(
        model, records, device=torch.device("cpu"), epochs=1, batch_size=8,
        optimizer=optimizer, value_weight=1.0, label="with-val", val_frac=0.25,
        val_seed=3,
    )
    assert len(no_val) == 1 and len(with_val) == 1
    assert math.isfinite(no_val[0]["loss"])
    assert "val_ce" not in no_val[0] and "val_mse" not in no_val[0]
    assert math.isfinite(with_val[0]["loss"])
    assert with_val[0]["val_ce"] > 0 and math.isfinite(with_val[0]["val_ce"])
    assert with_val[0]["val_mse"] >= 0 and math.isfinite(with_val[0]["val_mse"])


def main() -> None:
    test_flatten_records()
    test_records_to_tensors()
    test_losses_finite_and_backprop()
    test_train_epoch_updates_parameters()
    test_balanced_offsets()
    test_merge_rec_files()
    test_merge_rec_files_rejects_bad_streams()
    test_dihedral_transform_anchors()
    test_dihedral_transform_consistency()
    test_dihedral_transform_permutation()
    test_split_records()
    test_train_epoch_augmented_steps_parameters()
    test_train_on_records_val()
    print("OK: train.py pure-helper tests passed")


if __name__ == "__main__":
    main()
