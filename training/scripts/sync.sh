#!/usr/bin/env bash
# Hardware-aware torch sync for the training toolchain.
#
# Detects the machine (CPU-only vs NVIDIA GPU) via stdlib-only hardware.py and
# runs `uv sync` with the matching torch wheel index: a CPU box gets the small
# CPU build, a GPU box gets a compatible CUDA build. No manual --index juggling
# and no 4.6 GB CUDA download on a machine that cannot use it.
#
# Run inside `nix develop` (uv is a devshell package). Extra args are passed
# through to uv sync (e.g. --dry-run).
set -euo pipefail

# Resolve to this script's directory even when invoked via PATH (bare name).
cd "$(cd "$(dirname "$0")" && pwd)/.."

# The final `uv sync` is the point of the script, so uv is a hard requirement.
if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv is required (for the final 'uv sync') — run this inside 'nix develop'" >&2
  exit 1
fi

# hardware.py is stdlib-only, so any python works (torch is not needed yet).
PY=""
if command -v python3 >/dev/null 2>&1; then
  PY="python3"
elif command -v python >/dev/null 2>&1; then
  PY="python"
else
  PY="uv run --no-project python3"
fi

url="$($PY hardware.py)"
echo "Using torch index: $url"
# uv.lock is untracked because it encodes one hardware's torch (CPU vs CUDA).
# A stale lock would pin the wrong build (uv treats an existing lock as valid
# without re-resolving), so force a fresh hardware-specific resolve — unless the
# caller asked for a mode that must not touch the lock.
case " $* " in
  *" --dry-run "*|*" --frozen "*|*" --locked "*) ;;
  *) rm -f uv.lock ;;
esac
# shellcheck disable=SC2086  # PY is a word-split command
exec uv sync --index "$url" --index-strategy first-index "$@"
