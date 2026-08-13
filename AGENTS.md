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
The flake source is the **git-tracked** file set. The repo has no commits, so a
new file is invisible to `nix build` until registered:
```bash
git add -N path/to/new/file    # intent-to-add; makes `git ls-files` show it
```
Nothing is ever actually staged (`git diff --cached` must stay empty). After
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

## Workflow

- For any significant feature, use subagents: the worker to implement (from a
  plan in docs/plans/), then the reviewer and the worker in a review-revise loop
  until satisfied.
