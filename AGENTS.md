# AGENTS.md — gomoku

Offline Gomoku (five-in-a-row), player vs AI. C++23 + Qt 6 Widgets, Nix flake
toolchain. See `docs/REQUIREMENTS.md`, `docs/ARCHITECTURE.md`, `docs/plans/PLAN.md`.

## Build & test (do everything inside the devshell)
`g++` is NOT on PATH outside Nix. Always prefix with `nix develop -c`:
```bash
nix develop -c cmake -S . -B build -G Ninja
nix develop -c cmake --build build       # -Wall -Wextra -Werror on all targets
nix develop -c ctest --test-dir build    # 75 tests: 74 Catch2 + 1 Qt Test
nix flake check                          # builds the package (checkPhase = ctest)
nix run .#
```

## Nix + git gotcha (important)
The flake source is the **git-tracked** file set, so a new file is invisible to
`nix build` / `nix flake check` until registered with intent-to-add:
```bash
git add -N path/to/new/file    # intent-to-add; makes `git ls-files` show it
```
Nothing is ever actually staged (`git diff --cached` must stay empty). The
one-time `git rm --cached training/uv.lock` (untracking the hardware-specific
uv lock) is the sole allowed staged change; it is committed immediately. After
creating ANY new source file, run `git add -N` on it before `nix build` /
`nix flake check`, or the sandbox build fails with "Cannot find source file".

## Test-target structure (CMake)
- `gomoku_core` — Qt-free engine (Board, WinDetector, PatternTable, Evaluator,
  SearchEngine). Tested by `gomoku_tests` (Catch2, `catch_discover_tests`).
- `gomoku` — Qt Widgets app (main.cpp + src/ui/*).
- `gomoku_controller_tests` — Qt Test (`QTEST_GUILESS_MAIN`) for the
  `GameController` state machine. **GameController.cpp is compiled into the
  `gomoku` target, not `gomoku_core`**, so this test must list
  `src/ui/GameController.{hpp,cpp}` in its own sources and link `Qt6::Test`.
  Uses a real worker thread + event loop (`QTRY_VERIFY_WITH_TIMEOUT`).

## Known limitations (deliberately deferred)
- `GameController` joins the AI worker thread on the UI thread on New Game /
  shutdown, freezing the UI for up to `timeBudgetMs` (≤ 2 s at Hard). Correct
  and race-free; avoided a detach + `QPointer` rework (use-after-free risk).
- `AlphaBetaEngine::orderMoves` copies the `Board` per candidate move at every
  node; the time budget caps the cost. Could probe with in-place place/undo.

## Conventions
- All game rules and AI logic live in `src/core` and must stay Qt-free
  (`#include <Q...>` is forbidden there) so they stay Catch2-hermetic.
- AI runs on a background thread over a `Board` snapshot; results return via
  `Qt::QueuedConnection` and are dropped if the `aiEpoch_` no longer matches.

## Stage 3 training toolchain (Python, `training/`)
- `training/` is a `uv` project (`[tool.uv] package = false`), venv at
  `training/.venv` (gitignored + flake-excluded). Deps: torch, numpy.
- **Install torch with `training/scripts/sync.sh`** — it auto-detects the machine
  (CPU vs NVIDIA) and picks the right wheel index (small CPU build, or a CUDA
  build compatible with the driver). Bare `uv sync` pulls the default PyPI torch
  (~4.6 GB CUDA deps); `uv.lock` is **untracked** because it encodes one
  hardware's torch. At runtime use `hardware.select_device()` (cuda/cpu).
  Hardware logic: `training/hardware.py` (stdlib-only, hermetic test in
  `tests/test_hardware.py`).
- `.gnn` byte layout: `src/core/Weights.cpp` is the source of truth; Python
  mirror is `training/gnn_format.py` (stdlib-only, **encode only** — the trainer
  keeps the model in memory and only exports; it never loads a `.gnn` back into
  PyTorch).
- Training-data streams from `gomoku-bootstrap`/`gomoku-selfplay` use the
  `GameRecord` binary format (`src/core/GameRecord.cpp`); the Python side is
  `training/game_record.py` (stdlib-only decoder: magic `GOMOKUREC`, u32
  version=1, then per game `u32 numPositions` + `f32 planes[900] + f32
  policy[225] + f32 value`), gated by `tests/test_game_record.py` against real
  `gomoku-bootstrap` output.
- Inc 3's trainer (`training/train.py`) shells out to the C++ `gomoku-bootstrap`
  / `gomoku-selfplay` binaries (built in `build/`): bootstrap (SL) must come
  first, then `export_gnn` → `gomoku-selfplay --model X.gnn` (RL self-play needs
  a trained model) → train. The model stays in memory (only exported; `.gnn` is
  never reloaded into torch). Runtime device via `hardware.select_device()`.
  `--jobs J` parallelizes the single-threaded generators (J processes, byte-
  merged `.rec` streams).
- **Model shipping (Inc 4)**: the trained model is committed at
  `resources/model.gnn` and installed to `share/gomoku/model.gnn`. `flake.nix`
  whitelists the exact path `resources/model.gnn` (the source filter otherwise
  drops every `*.gnn`). `MainWindow::loadModel` loads `GOMOKU_MODEL_PATH`
  (override) else the bundled model, so the Neural engine works out of the box.
  Do NOT let generated `.gnn`/`.rec` files (training artifacts under `.pi/`) be
  committed — they stay git-ignored and flake-excluded.
- Head-to-head gate: `src/tools/headtohead.cpp` (`gomoku-headtohead`) plays the
  neural (MCTS) engine vs Classic-Hard alpha-beta; `tests/test_headtohead.py`
  drives it (report-only unless `--min-winrate` is set).
- Gate scripts (developer scripts, NOT ctest):
  ```bash
  cd training
  ./scripts/sync.sh                                   # hardware-aware torch install
  uv run python tests/test_hardware.py                # hermetic, no torch needed
  uv run python tests/test_gnn_roundtrip.py --tool ../build/gomoku-dump-weights  # stdlib-only
  uv run python tests/test_forward.py --tool ../build/gomoku-nn-eval             # needs torch
  uv run python tests/test_game_record.py --tool ../build/gomoku-bootstrap       # decoder gate
  uv run python tests/test_train.py                                              # hermetic torch helpers
  uv run python tests/test_headtohead.py --tool ../build/gomoku-headtohead \
      --model <model.gnn> --games N                                              # neural vs classic
  ```
- When the CUDA index list changes, run `uv run python tests/test_index_resolve.py`
  (network-gated: dry-resolves every listed torch index via `uv sync --dry-run`).

## Workflow

- For any significant feature, use subagents: the worker to implement (from a
  plan in docs/plans/), then the reviewer and the worker in a review-revise loop
  until satisfied.
- Use `uv` to manage Python dependencies, use `flake.nix` to manage other
  dependencies including `uv` itself.
