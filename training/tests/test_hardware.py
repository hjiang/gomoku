#!/usr/bin/env python3
"""Hermetic test for training/hardware.py (Stage 3 hardware-aware torch).

The index-mapping and GPU-detection logic is tested with injected ``GPUInfo``
objects and a fake ``run`` (canned nvidia-smi output) — no real GPU, network,
or torch needed. ``select_device`` does need torch (a venv dependency); it
asserts a ``torch.device`` is returned and that it is ``cpu`` when no GPU is
present.

Usage:  python tests/test_hardware.py
"""

import pathlib
import sys
import types

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import hardware

CPU_URL = hardware.CPU_INDEX_URL

FAKE_GPU_LISTING = (
    "GPU 0: NVIDIA GeForce RTX 4090 (UUID: GPU-00000000-0000-0000-0000-000000000000)\n"
)
FAKE_HEADER = (
    "Tue Aug 13 14:00:00 2026\n"
    "+---------------------------------------------------------------------------------------+\n"
    "| NVIDIA-SMI 550.54.15    Driver Version: 550.54.15    CUDA Version: 12.4     |\n"
    "+---------------------------------------------------------------------------------------+\n"
)


def url_for(name: str) -> str:
    for idx_name, _version, url in hardware.CUDA_INDEXES:
        if idx_name == name:
            return url
    raise AssertionError(f"no CUDA index named {name!r} in hardware.CUDA_INDEXES")


def gpu(cuda_version: str | None) -> hardware.GPUInfo:
    return hardware.GPUInfo(available=True, name="Fake GPU", cuda_version=cuda_version)


# --- torch_index_url: injected GPUInfo ---------------------------------------
def test_torch_index_url():
    cases = [
        # (driver CUDA version, expected index/fallback)
        ("13.0", url_for("cu130")),
        ("13.5", url_for("cu130")),
        ("12.8", url_for("cu128")),
        ("12.6", url_for("cu118")),  # cu128 (12.8) too new -> cu118
        ("12.4", url_for("cu118")),
        ("12.1", url_for("cu118")),
        ("11.8", url_for("cu118")),
        ("11.0", CPU_URL),   # too old for every CUDA index
        ("abc", CPU_URL),    # unparsable -> CPU fallback
        ("12", CPU_URL),     # missing minor -> CPU fallback
        (None, CPU_URL),     # GPU but no CUDA version -> CPU fallback
    ]
    for version, expected in cases:
        got = hardware.torch_index_url(gpu(version))
        assert got == expected, f"cuda {version!r}: got {got!r}, expected {expected!r}"

    # Explicit None means "no GPU".
    assert hardware.torch_index_url(None) == CPU_URL
    # available=False means "no GPU".
    assert hardware.torch_index_url(hardware.GPUInfo()) == CPU_URL

    # The mapping list is internally consistent: newest-first and every entry
    # under the pytorch download host.
    for i, (name, version, url) in enumerate(hardware.CUDA_INDEXES):
        if i + 1 < len(hardware.CUDA_INDEXES):
            assert version > hardware.CUDA_INDEXES[i + 1][1], f"{name} not newest-first"
        assert url.startswith("https://download.pytorch.org/whl/"), url


def test_torch_index_url_auto_detect():
    # The no-arg path auto-detects via detect_gpu; monkeypatch it to a fixed GPU.
    original = hardware.detect_gpu
    try:
        hardware.detect_gpu = lambda: gpu("12.4")
        assert hardware.torch_index_url() == url_for("cu118")
    finally:
        hardware.detect_gpu = original


# --- detect_gpu: injected fake run --------------------------------------------
def fake_run(listing_result, header_result):
    def run(argv, **kwargs):
        if argv == ["nvidia-smi", "-L"]:
            return listing_result
        if argv == ["nvidia-smi"]:
            return header_result
        raise AssertionError(f"unexpected command: {argv}")
    return run


def result(returncode=0, stdout="", stderr=""):
    return types.SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


def test_detect_gpu_gpu_present():
    run = fake_run(
        result(stdout=FAKE_GPU_LISTING),
        result(stdout=FAKE_HEADER),
    )
    info = hardware.detect_gpu(run=run)
    assert info.available is True
    assert info.name == "NVIDIA GeForce RTX 4090"
    assert info.cuda_version == "12.4"


def test_detect_gpu_no_devices():
    run = fake_run(
        result(returncode=1, stdout="No devices were found\n"),
        result(),
    )
    info = hardware.detect_gpu(run=run)
    assert info.available is False
    assert info.name is None
    assert info.cuda_version is None


def test_detect_gpu_no_nvidia_smi():
    def run(argv, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    info = hardware.detect_gpu(run=run)
    assert info.available is False


def test_detect_gpu_header_without_cuda_version():
    run = fake_run(
        result(stdout=FAKE_GPU_LISTING),
        result(stdout="NVIDIA-SMI 550.54.15    Driver Version: 550.54.15\n"),
    )
    info = hardware.detect_gpu(run=run)
    assert info.available is True
    assert info.name == "NVIDIA GeForce RTX 4090"
    assert info.cuda_version is None


def test_detect_gpu_empty_listing():
    # nvidia-smi -L exits 0 but lists no devices -> not available.
    run = fake_run(
        result(returncode=0, stdout=""),
        result(),
    )
    info = hardware.detect_gpu(run=run)
    assert info.available is False
    assert info.name is None
    assert info.cuda_version is None


def test_detect_gpu_header_oserror():
    # -L works, but the plain header call fails -> still a GPU, CUDA unknown.
    listing = result(stdout=FAKE_GPU_LISTING)

    def run(argv, **kwargs):
        if argv == ["nvidia-smi", "-L"]:
            return listing
        raise OSError("nvidia-smi header call failed")

    info = hardware.detect_gpu(run=run)
    assert info.available is True
    assert info.name == "NVIDIA GeForce RTX 4090"
    assert info.cuda_version is None


# --- select_device -------------------------------------------------------------
def test_select_device_returns_torch_device():
    import torch

    dev = hardware.select_device()
    assert isinstance(dev, torch.device)
    if not hardware.detect_gpu().available:
        assert dev.type == "cpu", f"expected cpu on a GPU-less machine, got {dev.type}"


def main():
    test_torch_index_url()
    test_torch_index_url_auto_detect()
    test_detect_gpu_gpu_present()
    test_detect_gpu_no_devices()
    test_detect_gpu_no_nvidia_smi()
    test_detect_gpu_header_without_cuda_version()
    test_detect_gpu_empty_listing()
    test_detect_gpu_header_oserror()
    test_select_device_returns_torch_device()
    print("OK: hardware selection tests passed")


if __name__ == "__main__":
    main()
