#!/usr/bin/env python3
"""Round-trip gate (stdlib-only): Python-written ``.gnn`` bytes == C++ loader.

Part of Stage 3 Increment 2. Builds a small model layout (N=1, C=2) where EVERY
element of EVERY tensor is filled with a unique per-name oracle value (an exact
float32, hardcoded below, independent of ``gnn_format``), then:

  (a) checks the raw bytes: header fields, total size, and a few specific
      byte probes at non-zero element offsets (pins offset math + little-endian
      encoding), and
  (b) runs ``gomoku-dump-weights`` and asserts the named dump reproduces the
      oracle value under every tensor name, in exactly the hardcoded
      EXPECTED_ORDER (the C++/``Weights.cpp`` order).

The check is NON-CIRCULAR: the oracle maps names to values and the expected
order is hardcoded, so a swap of two same-size tensors between ``gnn_format.py``
and the C++ loader is detected (the C++ dump would report a name carrying the
wrong oracle value, and the name sequence would deviate from EXPECTED_ORDER).

Usage:  python tests/test_gnn_roundtrip.py --tool ../build/gomoku-dump-weights
"""

import argparse
import pathlib
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import gnn_format

N_BLOCKS = 1
N_CHANNELS = 2

# Independent oracle: the exact order Weights.cpp reads tensors, hardcoded here
# (mirrors src/core/Weights.cpp). With N=1 every block tensor name is unique, so
# the list below covers all 28 tensor names exactly once.
EXPECTED_ORDER = [
    "inputConvW", "inputConvB",
    "inputBnGamma", "inputBnBeta", "inputBnMean", "inputBnVar",
    "conv1W", "conv1B",
    "bn1Gamma", "bn1Beta", "bn1Mean", "bn1Var",
    "conv2W", "conv2B",
    "bn2Gamma", "bn2Beta", "bn2Mean", "bn2Var",
    "policyConvW", "policyConvB",
    "policyFcW", "policyFcB",
    "valueConvW", "valueConvB",
    "valueFc1W", "valueFc1B",
    "valueFc2W", "valueFc2B",
]

# Each tensor gets a unique, exactly-representable float32 (0.0 .. 27.0). The
# dict is keyed by NAME, not derived from gnn_format's order.
NAME_VALUE = {name: float(i) for i, name in enumerate(EXPECTED_ORDER)}

# (tensor-name, element-index) probes at NON-ZERO element offsets: the raw bytes
# at these positions must decode to the tensor's oracle value, pinning the
# offset math and little-endian encoding.
BYTE_PROBES = [
    ("inputConvW", 71),
    ("inputConvB", 1),
    ("inputBnMean", 0),
    ("conv1W", 5),
    ("bn2Var", 1),
    ("policyConvB", 1),
    ("policyFcB", 7),
    ("valueConvB", 0),
    ("valueFc1B", 100),
    ("valueFc2B", 0),
]


def build_stream():
    """Return (data, order, index_by_name) with every element oracle-filled."""
    order = gnn_format.tensor_order(N_BLOCKS, N_CHANNELS)
    index_by_name = {name: i for i, (name, _) in enumerate(order)}
    tensors = [[NAME_VALUE[name]] * n for name, n in order]
    data = gnn_format.serialize(
        N_BLOCKS, N_CHANNELS, (gnn_format.pack_f32s(t) for t in tensors)
    )
    return data, order, index_by_name


def check_bytes(data, index_by_name):
    """(a) Raw bytes: header, total size, and oracle probes at known offsets."""
    assert data[:9] == gnn_format.MAGIC
    magic, version, num_blocks, channels = gnn_format.HEADER.unpack_from(data, 0)
    assert magic == gnn_format.MAGIC
    assert version == gnn_format.VERSION
    assert (num_blocks, channels) == (N_BLOCKS, N_CHANNELS)

    expected_size = (
        gnn_format.HEADER_SIZE + gnn_format.total_floats(N_BLOCKS, N_CHANNELS) * 4
    )
    assert len(data) == expected_size, (len(data), expected_size)

    offsets = gnn_format.tensor_offsets(N_BLOCKS, N_CHANNELS)
    for name, element in BYTE_PROBES:
        ti = index_by_name[name]
        off = offsets[ti] + element * 4
        got = struct.unpack_from("<f", data, off)[0]
        want = NAME_VALUE[name]
        assert got == want, (
            f"{name}[{element}] raw bytes decoded to {got!r}, expected oracle {want!r}"
        )
    print(f"[bytes] header, size ({expected_size}B), {len(BYTE_PROBES)} probes OK")


def parse_named_dump(lines):
    """Parse the named dump into [(name, n_floats, [floats]), ...] in C++ order."""
    sections = []
    name, n, floats = None, 0, []

    def flush():
        nonlocal name, n, floats
        if name is not None:
            sections.append((name, n, floats))
        name, n, floats = None, 0, []

    for line in lines:
        parts = line.split()
        if len(parts) == 2 and parts[0] in NAME_VALUE:
            flush()
            name = parts[0]
            n = int(parts[1])
            floats = []
        elif parts and parts[0] in ("num_blocks", "channels"):
            continue
        else:
            floats.append(float(line))
    flush()
    return sections


def check_cpp_loader(tool, data):
    """(b) gomoku-dump-weights reproduces the oracle under every name, in order."""
    with tempfile.TemporaryDirectory() as td:
        gnn = pathlib.Path(td) / "model.gnn"
        gnn.write_bytes(data)
        proc = subprocess.run([tool, "--model", str(gnn)], capture_output=True, text=True)
    assert proc.returncode == 0, f"gomoku-dump-weights failed: {proc.stderr.strip()}"

    lines = proc.stdout.splitlines()
    assert lines[0] == f"num_blocks {N_BLOCKS}", lines[0]
    assert lines[1] == f"channels {N_CHANNELS}", lines[1]

    sections = parse_named_dump(lines)
    names = [nm for nm, _, _ in sections]
    assert names == EXPECTED_ORDER, f"C++ dump order {names} != expected {EXPECTED_ORDER}"

    total = 0
    for nm, n, floats in sections:
        assert len(floats) == n, f"{nm}: dump has {len(floats)} floats, header says {n}"
        for v in floats:
            assert v == NAME_VALUE[nm], f"{nm}: value {v!r} != oracle {NAME_VALUE[nm]!r}"
        total += n
    assert total == gnn_format.total_floats(N_BLOCKS, N_CHANNELS), total

    print(f"[cpp] loader round-trips {total} floats across {len(sections)} named "
          f"tensors; order + oracle values OK")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool", required=True, help="path to gomoku-dump-weights")
    args = parser.parse_args()

    data, order, index_by_name = build_stream()
    check_bytes(data, index_by_name)
    check_cpp_loader(args.tool, data)
    print("OK: stdlib round-trip gate passed")


if __name__ == "__main__":
    main()
