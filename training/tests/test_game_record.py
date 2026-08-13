#!/usr/bin/env python3
"""Gate for the Python GameRecord decoder (``training/game_record.py``).

Three layers:

  (a) **Hermetic** — a record stream is built byte-by-byte with ``struct``
      (independent of ``game_record``), decoded, and the values asserted.
      Malformed inputs (bad magic, bad version, truncation, an oversized
      ``numPositions``, trailing bytes) must each raise ``ValueError``.
  (b) **Round-trip** — ``encode_stream(decode_stream(x)) == x`` and
      decode -> encode -> decode is idempotent.
  (c) **Integration** — runs ``gomoku-bootstrap`` (via ``--tool``) and decodes
      its output, asserting the stream is well-formed (plane/policy sizes,
      binary planes, policy sums to 1, value in {-1,0,1}).

Stdlib-only: no torch, no numpy.

Usage:  python tests/test_game_record.py [--tool ../build/gomoku-bootstrap]
"""

import argparse
import pathlib
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import game_record

MAGIC = game_record.MAGIC
VERSION = game_record.VERSION
PLANES = game_record.PLANES
POLICY = game_record.POLICY

# Independent mirrors (must equal game_record's constants; pins drift).
assert MAGIC == b"GOMOKUREC"
assert VERSION == 1
assert PLANES == 900
assert POLICY == 225
assert game_record.HEADER_SIZE == 13
assert game_record.POSITION_FLOATS == 1126
assert game_record.POSITION_BYTES == 4504


def _pos_bytes(plane_val: float, value: float) -> bytes:
    """One position record as raw little-endian bytes (struct only)."""
    planes = [float(plane_val)] * PLANES
    policy = [0.0] * POLICY
    policy[0] = 0.25  # exactly representable; sums to 1 with policy[1]
    policy[1] = 0.75
    return (
        struct.pack(f"<{PLANES}f", *planes)
        + struct.pack(f"<{POLICY}f", *policy)
        + struct.pack("<f", value)
    )


def _stream() -> bytes:
    """A two-game stream built entirely with struct (non-circular)."""
    out = bytearray(MAGIC)
    out += struct.pack("<I", VERSION)
    out += struct.pack("<I", 2)  # game 0: two positions
    out += _pos_bytes(1.0, 1.0)
    out += _pos_bytes(2.0, -1.0)
    out += struct.pack("<I", 1)  # game 1: one position
    out += _pos_bytes(3.0, 0.0)
    return bytes(out)


# --- (a) hermetic decode ------------------------------------------------------
def test_decode_values() -> None:
    games = game_record.decode_stream(_stream())
    assert len(games) == 2
    assert [len(g) for g in games] == [2, 1]

    p0 = games[0][0]
    assert p0.value == 1.0
    assert len(p0.planes) == PLANES
    assert len(p0.policy) == POLICY
    assert all(v == 1.0 for v in p0.planes)
    assert p0.policy[0] == 0.25
    assert p0.policy[1] == 0.75

    p1 = games[0][1]
    assert p1.value == -1.0
    assert all(v == 2.0 for v in p1.planes)

    p2 = games[1][0]
    assert p2.value == 0.0
    assert all(v == 3.0 for v in p2.planes)


def test_decode_empty_stream() -> None:
    games = game_record.decode_stream(MAGIC + struct.pack("<I", VERSION))
    assert games == []


def test_malformed_streams_raise() -> None:
    s = _stream()
    cases = [
        (b"X" + s[1:], "bad magic"),
        (s[:9] + struct.pack("<I", 2) + s[13:], "unsupported version"),
        (s[:5], "truncated header"),
        (s[:-8], "truncated position"),
        (MAGIC + struct.pack("<I", VERSION) + struct.pack("<I", 0xFFFFFFFF),
         "oversized numPositions"),
        (s + b"\x00", "trailing byte"),
    ]
    for data, label in cases:
        try:
            game_record.decode_stream(data)
        except ValueError:
            continue
        raise AssertionError(f"{label}: expected ValueError")


# --- (b) round-trip -----------------------------------------------------------
def test_round_trip() -> None:
    s = _stream()
    games = game_record.decode_stream(s)
    assert game_record.encode_stream(games) == s

    # decode -> encode -> decode is idempotent.
    again = game_record.decode_stream(game_record.encode_stream(games))
    assert len(again) == len(games)
    for g, g2 in zip(games, again):
        assert len(g2) == len(g)
        for p, p2 in zip(g, g2):
            assert p.planes == p2.planes
            assert p.policy == p2.policy
            assert p.value == p2.value


# --- (c) integration vs gomoku-bootstrap --------------------------------------
def test_bootstrap_output(tool: str) -> None:
    with tempfile.TemporaryDirectory() as td:
        out = pathlib.Path(td) / "bootstrap.rec"
        proc = subprocess.run(
            [tool, "--games", "3", "--depth", "1", "--time-ms", "2000",
             "--seed", "1", "--out", str(out)],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, f"gomoku-bootstrap failed: {proc.stderr.strip()}"

        games = game_record.decode_file(str(out))
        assert len(games) >= 1
        total_pos = sum(len(g) for g in games)
        assert total_pos >= 1

        for game in games:
            assert len(game) >= 1
            for pos in game:
                assert len(pos.planes) == PLANES
                assert len(pos.policy) == POLICY
                assert all(v in (0.0, 1.0) for v in pos.planes), "planes must be binary"
                policy_sum = sum(pos.policy)
                assert abs(policy_sum - 1.0) < 1e-3, f"policy sums to {policy_sum}"
                assert pos.value in (-1.0, 0.0, 1.0), f"value {pos.value!r}"
    print(f"[integration] gomoku-bootstrap output decoded: {len(games)} games, "
          f"{total_pos} positions, all well-formed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool", required=True, help="path to gomoku-bootstrap")
    args = parser.parse_args()

    test_decode_values()
    test_decode_empty_stream()
    test_malformed_streams_raise()
    test_round_trip()
    test_bootstrap_output(args.tool)
    print("OK: GameRecord decoder gate passed")


if __name__ == "__main__":
    main()
