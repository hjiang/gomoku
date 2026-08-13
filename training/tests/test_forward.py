#!/usr/bin/env python3
"""C++ <-> PyTorch forward-pass gate (Stage 3 Increment 2).

Builds a GomokuNet (N=4, C=16) with deterministic random tensors — including
BatchNorm running_mean/running_var — exports it to ``.gnn`` via export_gnn,
evaluates a deterministic random input in PyTorch (eval mode) and in C++
(gomoku-nn-eval), and asserts the 225 policy logits and the value agree within
relative tolerance 1e-4 (absolute floor 1e-5).

Usage:  python tests/test_forward.py --tool ../build/gomoku-nn-eval
"""

import argparse
import math
import pathlib
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch

import export_gnn
import model as model_mod

N_BLOCKS = 4
N_CHANNELS = 16


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool", required=True, help="path to gomoku-nn-eval")
    args = parser.parse_args()

    torch.manual_seed(20260813)
    # This gate compares against the C++ engine, which runs on CPU only, so
    # PyTorch must stay on CPU here regardless of host hardware (never .to("cuda")).
    net = model_mod.GomokuNet(num_blocks=N_BLOCKS, channels=N_CHANNELS)
    # Small deterministic random weights keep activations ~O(1) (a real trained
    # network lives in this range). With magnitudes ~1e4 float32 accumulation
    # order between the naive C++ loop and BLAS shows up at ~1e-4 relative, so
    # an adversarial init would mask real architecture/layout bugs.
    for module in net.modules():
        if isinstance(module, torch.nn.Conv2d):
            torch.nn.init.uniform_(module.weight, -0.05, 0.05)
            torch.nn.init.uniform_(module.bias, -0.05, 0.05)
        elif isinstance(module, torch.nn.Linear):
            torch.nn.init.uniform_(module.weight, -0.02, 0.02)
            torch.nn.init.uniform_(module.bias, -0.05, 0.05)
        elif isinstance(module, torch.nn.BatchNorm2d):
            torch.nn.init.uniform_(module.weight, 0.8, 1.2)  # gamma
            torch.nn.init.uniform_(module.bias, -0.1, 0.1)   # beta
            with torch.no_grad():
                module.running_mean.normal_(mean=0.0, std=0.05)
                module.running_var.uniform_(0.8, 1.2)
    net.eval()

    # Deterministic random input in [-1, 1]: (1, 4, 15, 15), channels-first.
    g = torch.Generator().manual_seed(424242)
    x = torch.rand(1, 4, 15, 15, generator=g) * 2.0 - 1.0

    with torch.no_grad():
        logits_t, value_t = net(x)
    logits_t = [float(v) for v in logits_t[0]]
    value_t = value_t.item()  # (1, 1) -> scalar

    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        gnn = td / "model.gnn"
        gnn.write_bytes(export_gnn.export_gnn(net))
        planes = td / "input.planes"
        planes.write_bytes(x[0].contiguous().numpy().astype("<f4").tobytes())
        proc = subprocess.run(
            [args.tool, "--model", str(gnn), "--input", str(planes)],
            capture_output=True,
            text=True,
        )
    assert proc.returncode == 0, f"gomoku-nn-eval failed: {proc.stderr.strip()}"

    lines = proc.stdout.splitlines()
    assert len(lines) == 225 + 1, f"expected 226 output lines, got {len(lines)}"
    logits_c = [float(v) for v in lines[:225]]
    value_c = float(lines[225])

    max_policy_rel = 0.0
    for i, (a, b) in enumerate(zip(logits_t, logits_c)):
        ok = math.isclose(a, b, rel_tol=1e-4, abs_tol=1e-5)
        max_policy_rel = max(max_policy_rel, abs(a - b) / max(abs(a), abs(b), 1e-30))
        assert ok, f"policy logit {i}: torch={a!r} cpp={b!r}"

    value_rel = abs(value_t - value_c) / max(abs(value_t), abs(value_c), 1e-30)
    assert math.isclose(value_t, value_c, rel_tol=1e-4, abs_tol=1e-5), (
        f"value: torch={value_t!r} cpp={value_c!r}"
    )
    print(f"OK: policy max rel err {max_policy_rel:.3e}; value rel err {value_rel:.3e}")


if __name__ == "__main__":
    main()
