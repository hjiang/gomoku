#!/usr/bin/env python3
"""Resolvability gate for the torch indexes listed in ``training/hardware.py``.

Network-gated DEVELOPER script — NOT hermetic and NOT part of the normal gate
suite: for the CPU index plus every CUDA index it creates a throwaway uv
project mirroring the real pyproject (``torch>=2.7``, ``numpy``,
``requires-python >=3.11,<3.14``) and runs ``uv sync --dry-run --index <url>
--index-strategy first-index``, asserting the resolve succeeds. Run it manually
whenever the index list changes:

    uv run python tests/test_index_resolve.py

Requires network access to download.pytorch.org and the ``uv`` binary.
"""

import pathlib
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import hardware

# Mirrors training/pyproject.toml's dependency spec (the floor and the Python
# bound), so a change to either is reflected here.
PROJECT_PYPROJECT = """\
[project]
name = "index-check"
version = "0.1.0"
requires-python = ">=3.11,<3.14"
dependencies = ["torch>=2.7", "numpy"]

[tool.uv]
package = false
"""


def check_index(url: str) -> tuple[bool, str]:
    """Dry-resolve ``url``; returns ``(ok, detail)`` with the resolved torch or
    the failure reason."""
    with tempfile.TemporaryDirectory() as td:
        pathlib.Path(td, "pyproject.toml").write_text(PROJECT_PYPROJECT)
        proc = subprocess.run(
            ["uv", "sync", "--dry-run", "--index", url, "--index-strategy", "first-index"],
            cwd=td,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            return False, proc.stderr.strip() or "(no stderr)"
        # uv writes the resolved package list to stderr (e.g. " + torch==2.13.0+cpu").
        for stream in (proc.stderr, proc.stdout):
            for line in stream.splitlines():
                if "torch==" in line:
                    return True, line.split("torch==", 1)[1].strip()
        return False, "resolved but no 'torch==' line in output"


def main() -> None:
    urls = [hardware.CPU_INDEX_URL] + [url for _, _, url in hardware.CUDA_INDEXES]
    failures = 0
    for url in urls:
        ok, detail = check_index(url)
        status = "OK" if ok else "FAIL"
        print(f"  {status} {url} -> {detail}")
        if not ok:
            failures += 1
    if failures:
        print(f"FAIL: {failures} index(es) did not resolve")
        sys.exit(1)
    print("OK: all torch indexes resolve")


if __name__ == "__main__":
    main()
