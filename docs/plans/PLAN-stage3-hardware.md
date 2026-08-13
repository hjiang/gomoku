# Plan — Stage 3 hardware-aware training toolchain

Parent: `PLAN-stage3-training.md`. Makes the `training/` toolchain automatically
pick the right torch build and runtime device for the machine it runs on (CPU-only
vs NVIDIA GPU), instead of hardcoding one path.

## Goal
On any machine, one command installs the correct torch (CUDA on a GPU box, tiny
CPU wheel otherwise) and the training code runs on the right device — no manual
`--index-url` juggling, no 4.6 GB CUDA download on a CPU box, no `CUDA error`
from an incompatible wheel.

## Facts (verified on the CPU machine, `uv 0.11.26`)
- Default PyPI `torch` (2.13.0) pulls ~18 `nvidia-*`/`cuda-*` packages (≈4.6 GB).
- `uv sync --index https://download.pytorch.org/whl/cpu --index-strategy first-index`
  resolves `torch==2.13.0+cpu` with only 11 packages, no nvidia deps.
- CUDA index → torch version skew is real: `cu130` → 2.13.0, `cu128` → 2.11.0,
  `cu118` → 2.7.1, `cpu` → 2.13.0. The `.gnn` model/export code is
  version-agnostic, so this is acceptable; document it. `cu121`/`cu124` were
  dropped: they do not resolve under `uv sync` (missing `nvidia-cudnn-cu12`
  pins / aarch64-only wheels). Drivers 11.8–12.7 fall through to `cu118`, whose
  binaries run on newer drivers via CUDA backward compatibility.
- `nvidia-smi` prints `CUDA Version: X.Y` (the driver's max supported CUDA
  runtime) — the authoritative cap for which CUDA wheel will run.

## Design

### `training/hardware.py` (stdlib-only, no torch import at module top)
- `GPUInfo` dataclass: `available: bool`, `name: str | None`, `cuda_version:
  str | None` (e.g. `"12.4"`).
- `detect_gpu(run=subprocess.run) -> GPUInfo` — runs `nvidia-smi` (injectable
  `run` for hermetic tests); `available` = `nvidia-smi -L` exits 0 and lists a
  GPU; parses the `CUDA Version: X.Y` header line.
- `CUDA_INDEXES` — ordered list of `(name, version_tuple, url)`:
  `cu130 (13,0)`, `cu128 (12,8)`, `cu118 (11,8)` (all under
  `https://download.pytorch.org/whl/`). Newest-first; every entry must resolve
  via the network-gated gate (below).
- `torch_index_url(gpu: GPUInfo | None = None) -> str` — CPU url when no GPU;
  otherwise the highest CUDA index whose version ≤ the driver's; CPU url (with
  reason) when the driver is too old or unparsable.
- `select_device() -> torch.device` — lazy `import torch`; returns
  `cuda` if `torch.cuda.is_available()` else `cpu`, and warns when a GPU is
  present but torch has no CUDA (→ re-run the sync script).
- `__main__` prints `torch_index_url()` (used by the sync script).

### `training/scripts/sync.sh` (the one entry point)
```bash
cd training
url="$(python3 hardware.py)"          # fall back to `python`, then `uv run --no-project python3`
echo "Using torch index: $url"
exec uv sync --index "$url" --index-strategy first-index "$@"
```
Passes through extra args (`--dry-run`, …). Run inside `nix develop`.

### Python interpreter
`requires-python = ">=3.11,<3.14"` plus `training/.python-version` = `3.11`:
the oldest supported index (`cu118`, torch 2.7.1) stops at cp313, so a newer
interpreter would make older-driver machines unresolvable.

### Lockfile
`training/uv.lock` becomes **gitignored and untracked**: it encodes one
hardware's torch wheel, so a committed lock would churn/conflict across machines.
`git rm --cached training/uv.lock` + add to `.gitignore`. Pin a floor in
`pyproject.toml` (`torch>=2.7`) so the float doesn't drift below the oldest CUDA
index's torch (cu118 serves 2.7.1).

### Test
`training/tests/test_hardware.py` — hermetic, stdlib-only assertions over
`torch_index_url` with injected `GPUInfo` (CPU → cpu url; 13.0 → cu130; 12.8 →
cu128; 12.6/12.4/12.1/11.8 → cu118; 11.0/None → cpu fallback) and `detect_gpu`
with a fake `run` (fake `nvidia-smi -L` output). `select_device` returns a
`torch.device` and is `cpu` here. Plain assert script, run via
`uv run python tests/test_hardware.py`.

### Resolvability gate (network-gated)
`training/tests/test_index_resolve.py` — a DEVELOPER script (not hermetic, not
ctest): for each listed index plus CPU it dry-resolves the exact `uv sync
--index <url> --index-strategy first-index` in a throwaway project and fails if
any index cannot be installed. Run it manually when the index list changes.

### Docs
- `docs/ARCHITECTURE.md` training section: hardware-aware sync + `select_device`.
- `AGENTS.md` Stage-3 section: replace the `--index-url cpu` line with "run
  `training/scripts/sync.sh` (auto-detects CPU vs NVIDIA); `select_device()` for
  the runtime device".
- `test_forward.py`: add an explicit CPU pin comment (it compares against the
  CPU C++ engine, so PyTorch must stay on CPU there regardless of hardware).

## Acceptance
- `python3 training/hardware.py` prints the CPU index on this machine.
- `training/tests/test_hardware.py` passes (hermetic).
- Existing gates + `ctest` + `nix flake check` stay green; `uv.lock` untracked.
- New files `git add -N`; `git diff --cached` empty.
