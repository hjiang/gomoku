# Plan — Stage 3, Increment 2: Python model + `.gnn` exporter (round-trip gate)

Parent: [`PLAN-stage3-training.md`](PLAN-stage3-training.md). Requirements FR13–FR15.
This increment adds the Python (PyTorch) half of the training toolchain and a
round-trip gate proving the Python and C++ sides agree on both the `.gnn` byte
format and the network's numeric behavior. No training loop yet (Increment 3).

## Goal
Produce a PyTorch model that is byte-for-byte layout-compatible with the C++
inference engine, an exporter that writes `*.gnn` files `Weights::parse` can
read, and two gate scripts that prove:
1. the `.gnn` **format contract** is shared (stdlib Python writes bytes → C++
   `Weights::parse` reads them back to the same values), and
2. the **numeric semantics** are shared (same weights + same input →
   C++ `NeuralNet::evaluate` and PyTorch forward agree within rel 1e-4).

## Architecture contract (authoritative)

The C++ is the single source of truth: `src/core/Weights.cpp` (tensor order /
byte layout), `src/core/NeuralNet.cpp` + `src/core/TensorOps.hpp` (forward
math), `src/core/BoardEncoder.cpp` (input planes). The Python side must mirror
these exactly.

### Input
4 binary planes of 15×15, **channels-first**: plane 0 = to-move stones,
plane 1 = opponent stones, plane 2 = constant 1, plane 3 = last-move marker.
`BoardEncoder::encode` produces exactly this (`kInputSize = 900`).

### Network (must match `NeuralNet::evaluate`)
- Input projection: `conv3×3(4 → C, pad 1) → BN → ReLU`.
- Body: `N` residual blocks, each
  `conv3×3(C, pad 1) → BN → ReLU → conv3×3(C, pad 1) → BN → add-input → ReLU`.
- Policy head: `conv3×3(C → 2, pad 1) → flatten(2·225 = 450, channel-major) →
  linear(450 → 225)` → raw logits (no softmax, no masking).
- Value head: `conv3×3(C → 1, pad 1) → flatten(225) → linear(225 → 256) →
  ReLU → linear(256 → 1) → tanh`.
- BatchNorm is **eval-mode running stats** (`track_running_stats=True`,
  `eps = 1e-5`).

PyTorch `nn.Conv2d(outC, inC, 3, padding=1)` weight is `(outC, inC, 3, 3)`;
its contiguous flatten is `o·(inC·9) + i·9 + dy·3 + dx`, which is exactly the
C++ `conv2d` weight index. `nn.Linear(in, out)` weight is `(out, in)`; its
flatten is `j·in + i`, matching C++ `linear`. A `(B, C, H, W)` tensor's
`.flatten(1)` is channel-major, matching C++ channel-major flattening.

### `.gnn` byte format (must match `Weights::parse`)
Header (little-endian): magic `"GOMOKUNET"` (9 bytes) + `u32 version = 1` +
`u32 numBlocks` + `u32 channels`. Then all tensors as little-endian f32, in
this exact order (block section repeated `numBlocks` times):

```
inputConvW   (C, 4, 3, 3)
inputConvB   (C)
inputBnGamma (C)      # BatchNorm.weight
inputBnBeta  (C)      # BatchNorm.bias
inputBnMean  (C)      # BatchNorm.running_mean
inputBnVar   (C)      # BatchNorm.running_var
for each block:
  conv1W (C, C, 3, 3)  conv1B (C)
  bn1Gamma (C) bn1Beta (C) bn1Mean (C) bn1Var (C)
  conv2W (C, C, 3, 3)  conv2B (C)
  bn2Gamma (C) bn2Beta (C) bn2Mean (C) bn2Var (C)
policyConvW (2, C, 3, 3)   policyConvB (2)
policyFcW   (225, 450)     policyFcB (225)
valueConvW  (1, C, 3, 3)   valueConvB (1)
valueFc1W   (256, 225)     valueFc1B (256)
valueFc2W   (1, 256)       valueFc2B (1)
```

## Deliverables

### Python (managed by `uv`, project in `training/`)
- `training/gnn_format.py` — **stdlib-only**, the single Python source of truth
  for the layout: `MAGIC`, `VERSION`, and an ordered tensor spec (name + shape
  with symbolic `"C"`/block repetition) plus a `serialize(num_blocks, channels,
  tensors)` helper. No torch import. Both `export_gnn.py` and the stdlib gate
  derive offsets from it, so the layout lives in exactly one Python place.
- `training/model.py` — `GomokuNet(nn.Module)` with `__init__(num_blocks=4,
  channels=16)`; explicit submodules `input_conv/input_bn`, `blocks`
  (`nn.ModuleList` of a `ResidualBlock` with `conv1/bn1/conv2/bn2`),
  `policy_conv/policy_fc`, `value_conv/value_fc1/value_fc2`. `forward(x)` on
  `(B, 4, 15, 15)` returns `(policy_logits (B, 225), value (B, 1))`, value =
  `tanh`, no softmax. All BatchNorm `eps=1e-5`, `track_running_stats=True`.
- `training/export_gnn.py` — `export_gnn(model) -> bytes` and a `--out FILE`
  CLI. Yields each tensor **by explicit attribute access** (not `state_dict()`
  key matching) in the exact order above, casts to f32, `contiguous()`, and
  writes little-endian. Header uses `model.num_blocks` / `model.channels`.
- `training/pyproject.toml` — uv project, `[tool.uv] package = false`,
  dependency `torch`.

### C++ developer tools (`src/tools/`, link `gomoku_core`, **never installed**)
- `src/tools/dump_weights.cpp` → `gomoku-dump-weights --model FILE.gnn`: parses
  with `Weights::parse` and prints `num_blocks <N>`, `channels <C>`, then every
  float in exact tensor order, one per line (`%.9g`, exact f32 round-trip).
- `src/tools/nn_eval.cpp` → `gomoku-nn-eval --model FILE.gnn --input FILE.planes`:
  reads the model, reads 900 little-endian f32 input planes, runs
  `NeuralNet::evaluate`, prints 225 policy logits then `value` (one per line).
- Register both in `CMakeLists.txt` (build, not install).

### Gate scripts (developer scripts, **not** ctest)
- `training/tests/test_gnn_roundtrip.py` — **stdlib-only** (struct only, no
  torch/numpy). Builds a small model layout (N=1, C=2) with a handful of known
  nonzero values at known offsets via `gnn_format`, writes the `.gnn`, (a)
  checks the raw bytes at the computed offsets equal the expected little-endian
  f32, and (b) runs `gomoku-dump-weights` and asserts the C++ loader reports the
  identical values (and zeros elsewhere). Passes the tool path as an argv/env.
- `training/tests/test_forward.py` — **needs torch**. Seeds a `GomokuNet`
  (N=4, C=16), fills every tensor — including `running_mean`/`running_var` —
  with deterministic random values, exports via `export_gnn.py`, generates a
  deterministic random input, writes it as 900 little-endian f32, runs
  `gomoku-nn-eval`, and asserts each policy logit and the value agree with
  `model.eval()(input)` within rel 1e-4 (with an absolute floor so near-zero
  entries don't spuriously fail).

### Tooling / housekeeping
- `flake.nix`: add `uv` to the devShell; extend the `cleanSourceWith` filter to
  exclude `.venv`, `__pycache__`, and `*.gnn`/`*.rec` so `nix flake check` never
  tries to copy the (huge) torch venv or generated training artifacts into the
  store.
- `.gitignore`: add `.venv/`, `__pycache__/`, `*.gnn`, `*.rec`.
- Update `docs/ARCHITECTURE.md` (new `training/` Python layer + the two tools +
  the gate scripts) and tick Increment 2 in `PLAN-stage3-training.md`.

## Test discipline (TDD)
Write the gate scripts first (they fail: no tools / no modules), confirm they
fail, then implement `gnn_format.py`, `model.py`, `export_gnn.py`, and the two
C++ tools until they pass. Keep existing `ctest` green throughout (`-Werror`).

## Verification commands
```bash
nix develop -c cmake -S . -B build -G Ninja
nix develop -c cmake --build build
nix develop -c ctest --test-dir build          # existing 75 tests stay green
nix develop -c ./build/gomoku-dump-weights --model /tmp/x.gnn   # smoke
cd training && uv sync                          # installs torch
uv run python tests/test_gnn_roundtrip.py --tool ../build/gomoku-dump-weights
uv run python tests/test_forward.py --tool ../build/gomoku-nn-eval
nix flake check
```

## Acceptance criteria
- All three Python modules + two gate scripts present; `test_gnn_roundtrip.py`
  (stdlib) and `test_forward.py` (torch) both pass.
- `Weights::parse(export_gnn.py output)` round-trips; C++/PyTorch forward agree
  within rel 1e-4.
- `nix develop -c ctest --test-dir build` green, `-Werror` clean,
  `nix flake check` passes.
- New files registered with `git add -N`; nothing staged (`git diff --cached`
  empty).
