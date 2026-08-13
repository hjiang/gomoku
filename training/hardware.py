"""Hardware-aware torch selection for the training toolchain.

Stdlib-only (no ``torch`` import at module top): GPU detection and the
CPU/CUDA torch index choice work *before* torch is installed, which is exactly
when the sync script needs them. The runtime device picker (``select_device``)
imports torch lazily, so it can be called after ``uv sync``.

Torch ships as separate builds per hardware:

- CPU-only wheel: ``https://download.pytorch.org/whl/cpu`` (small, no nvidia deps).
- CUDA wheels: ``https://download.pytorch.org/whl/cuXXX`` where ``cuXXX`` must be
  <= the NVIDIA driver's max supported CUDA runtime (reported by ``nvidia-smi``
  as ``CUDA Version: X.Y``).

Note the CUDA index -> torch version skew: ``cu130`` -> 2.13.0, ``cu128`` ->
2.11.0, ``cu118`` -> 2.7.1, ``cpu`` -> 2.13.0. Drivers 11.8-12.7 select ``cu118``
(torch 2.7.1), which runs on newer drivers via CUDA backward compatibility. The
`.gnn` model/export code is version-agnostic, so the skew is acceptable.
"""

from __future__ import annotations

import dataclasses
import functools
import re
import subprocess
import sys

# The CPU-only torch wheel index (works everywhere; no CUDA acceleration).
CPU_INDEX_URL = "https://download.pytorch.org/whl/cpu"

# Ordered newest-first CUDA torch indexes: (name, (major, minor), url). A wheel
# from index cuX.Y runs on any driver whose reported CUDA version is >= X.Y.
# cu121/cu124 were dropped: they do not resolve under `uv sync` (missing
# nvidia-cudnn-cu12 pins / aarch64-only wheels). Drivers 11.8-12.7 fall through
# to cu118; cu118 binaries run on newer drivers via CUDA backward compatibility.
CUDA_INDEXES = [
    ("cu130", (13, 0), "https://download.pytorch.org/whl/cu130"),
    ("cu128", (12, 8), "https://download.pytorch.org/whl/cu128"),
    ("cu118", (11, 8), "https://download.pytorch.org/whl/cu118"),
]

# nvidia-smi header line: "... NVIDIA-SMI 550.54.15 ... CUDA Version: 12.4 |"
_CUDA_VERSION_RE = re.compile(r"CUDA Version:\s*(\d+)\.(\d+)")

# Sentinel distinguishing "caller passed nothing" (auto-detect) from an explicit
# None (meaning "no GPU").
_AUTO = object()


@dataclasses.dataclass(frozen=True)
class GPUInfo:
    """What ``nvidia-smi`` tells us about the machine."""

    available: bool = False
    name: str | None = None
    cuda_version: str | None = None  # e.g. "12.4" — the driver's max CUDA runtime


def _parse_listing_name(line: str) -> str | None:
    # "GPU 0: NVIDIA GeForce RTX 4090 (UUID: GPU-...)" -> "NVIDIA GeForce RTX 4090"
    rest = line.split(":", 1)[1].strip() if ":" in line else line.strip()
    if not rest:
        return None
    return rest.split(" (", 1)[0].strip()


def detect_gpu(run=subprocess.run) -> GPUInfo:
    """Detect an NVIDIA GPU via ``nvidia-smi``.

    ``run`` is injectable for hermetic tests (same signature as
    ``subprocess.run`` with ``capture_output=True, text=True``). A GPU is
    present when ``nvidia-smi -L`` exits 0 and lists at least one device; the
    CUDA version is parsed from the plain ``nvidia-smi`` header (the driver's
    max supported CUDA runtime).
    """
    try:
        listing = run(["nvidia-smi", "-L"], capture_output=True, text=True)
    except OSError:  # nvidia-smi not installed
        return GPUInfo()
    if listing.returncode != 0:
        return GPUInfo()
    gpus = [ln for ln in listing.stdout.splitlines() if ln.strip()]
    if not gpus:
        return GPUInfo()

    cuda_version: str | None = None
    try:
        header = run(["nvidia-smi"], capture_output=True, text=True)
        if header.returncode == 0:
            m = _CUDA_VERSION_RE.search(header.stdout)
            if m:
                cuda_version = f"{int(m.group(1))}.{int(m.group(2))}"
    except OSError:
        pass

    return GPUInfo(available=True, name=_parse_listing_name(gpus[0]), cuda_version=cuda_version)


def torch_index_url(gpu=_AUTO) -> str:
    """Best torch wheel index for the hardware.

    ``gpu`` is an injected ``GPUInfo`` for tests; pass ``None`` to mean "no
    GPU" or omit it to auto-detect. Returns the CPU index when there is no GPU
    (or the driver is too old / unparsable), otherwise the highest CUDA index
    whose version is <= the driver's reported CUDA version.
    """
    if gpu is _AUTO:
        gpu = detect_gpu()
    if gpu is None or not gpu.available or not gpu.cuda_version:
        return CPU_INDEX_URL

    major, sep, minor = gpu.cuda_version.partition(".")
    if not sep or not minor:
        return CPU_INDEX_URL  # unparsable -> safe CPU fallback
    try:
        driver = (int(major), int(minor))
    except ValueError:
        return CPU_INDEX_URL

    for _name, version, url in CUDA_INDEXES:
        if version <= driver:
            return url
    return CPU_INDEX_URL  # driver older than every CUDA index


@functools.lru_cache(maxsize=1)
def _detect_gpu_cached() -> GPUInfo:
    """Cached auto-detection for ``select_device``: spawns ``nvidia-smi`` at
    most once per process (repeated device selection stays cheap)."""
    return detect_gpu()


def select_device():
    """Best ``torch.device`` for this machine.

    Prefers CUDA when the installed torch was built with it and a GPU is
    available; falls back to CPU. Warns when an NVIDIA GPU is present but torch
    reports CUDA unavailable (wrong build or incompatible driver).
    """
    import torch  # lazy: hardware.py is used before torch is installed

    if torch.cuda.is_available():
        return torch.device("cuda")
    if _detect_gpu_cached().available:
        print(
            "warning: NVIDIA GPU detected but torch reports CUDA unavailable — "
            "either the CPU-only torch build is installed (run "
            "training/scripts/sync.sh) or the driver is too old/incompatible",
            file=sys.stderr,
        )
    return torch.device("cpu")


def main() -> None:
    # stdout is exactly the index URL (the sync script consumes it).
    print(torch_index_url())


if __name__ == "__main__":
    main()
