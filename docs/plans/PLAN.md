# Plan — Gomoku (Player vs AI, C++23 + Qt 6 + Nix)

Full requirements: [`docs/REQUIREMENTS.md`](../REQUIREMENTS.md).
Architecture: [`docs/ARCHITECTURE.md`](../ARCHITECTURE.md).

## Goals
Ship an offline, single-player Gomoku game: human (Black) vs a pattern-heuristic
alpha-beta AI (White), built with modern C++ and Qt 6 Widgets, with the entire
toolchain pinned by a Nix flake.

## Deliverables
1. A Qt-free core engine with unit tests (the bulk of the value).
2. A Qt 6 Widgets UI that plays the game without freezing during AI turns.
3. A `flake.nix` that provides: `nix develop` (dev shell), `nix build` (package),
   `nix run` (launch), and `nix flake check` (tests).

---

## Phase 0 — Toolchain & skeleton (foundation)

**Goal:** `nix develop` + `cmake --build` + `ctest` all work, with one trivial
core function and one test proving the pipeline end-to-end.

Tasks:
1. Write `flake.nix` (see sketch below) and `.envrc` (with `use flake`), `.gitignore`.
2. Write `CMakeLists.txt` with the three targets (`gomoku_core`, `gomoku`, `gomoku_tests`)
   and CTest registration.
3. Add `src/core/types.hpp` (constants + `Player`/`Position`/`Move`) and a trivial
   `src/core/Board.cpp` stub.
4. Add `tests/test_board.cpp` with one test (empty board is not full).
5. Wire CTest so `nix flake check` runs it.

**Sketch — `flake.nix`:**
```nix
{
  description = "Offline Gomoku — player vs AI (C++23, Qt 6)";
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-25.05";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
      in {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            cmake ninja gcc14 gdb
            qt6.qtbase qt6.wrapQtAppsHook
            catch2_3
            clang-tools
          ];
        };
        packages.default = pkgs.stdenv.mkDerivation {
          pname = "gomoku";
          version = "0.1.0";
          src = self;
          nativeBuildInputs = with pkgs; [ cmake qt6.wrapQtAppsHook ];
          buildInputs = with pkgs; [ qt6.qtbase catch2_3 ];
          doCheck = true;
        };
        apps.default = { type = "app"; program = "${self.packages.${system}.default}/bin/gomoku"; };
        checks = { inherit (self.packages.${system}) default; };
      });
}
```
(Adjust the nixpkgs branch to the current stable at implementation time; pinning a
specific `rev` is preferable to a moving branch for reproducibility.)

**Acceptance:** `nix develop -c cmake -S . -B build -G Ninja && nix develop -c cmake --build build && nix develop -c ctest --test-dir build` all succeed; `nix flake check` passes.

---

## Phase 1 — Core game rules (test-driven)

**Goal:** complete, correct board + win/draw detection, fully unit-tested, still
Qt-free.

Tasks (each: write failing test → see it fail → implement → refactor):
1. `Board`: `place`, `undo`, `isEmpty`, `inBounds`, `isFull`, `winner`-guarding preconditions.
2. `WinDetector::winnerOf`: 5-in-a-row horizontal / vertical / both diagonals;
   overline (6+); no false positive on broken lines; edge cases (stone on board edge).
3. `WinDetector::findWinningLine`: returns the exact 5 cells (for UI highlight).
4. Draw: full board with no winner returns draw state.

**Acceptance:** `tests/test_board.cpp` + `tests/test_windetector.cpp` green;
no Qt included anywhere in `src/core`.

---

## Phase 2 — Heuristic evaluator

**Goal:** a correct, symmetric pattern-based board scorer.

Tasks:
1. `PatternTable`: patterns (`_XXXX_`, `XXXX_`, `_XXXX`, `_XXX_`, `XX_X`, `_OOO_`, …)
   mapped to scores; built once (lazy `static`).
2. `Evaluator::score(board, player)`: scan all 4 directions; sum pattern scores.
3. `Evaluator::score(board, Black) == -score(color-swapped board, White)` test
   (symmetry invariant).

**Acceptance:** `tests/test_evaluator.cpp` green; symmetry and monotonicity tests pass.

---

## Phase 3 — AI search engine

**Goal:** an AI that finds immediate wins, blocks open fours, and respects a time budget.

Tasks:
1. `SearchEngine` negamax + alpha-beta; terminal-score depth adjustment.
2. Candidate-move generation (Chebyshev distance ≤ 2 from existing stones).
3. Iterative deepening with a `SearchParams { maxDepth, timeBudgetMs }` budget.
4. Difficulty mapping (Easy=2/3, Medium=4, Hard=6 or time-based).
5. Deterministic tie-breaking (or a seedable RNG for variety).

**Acceptance** (all in `tests/test_engine.cpp`): plays its winning move immediately;
blocks an opponent's open four; returns within the time budget; returns the same move
for a fixed seed.

---

## Phase 4 — Qt 6 UI

**Goal:** a playable window; AI runs off the UI thread.

Tasks:
1. `BoardWidget` custom `QWidget`: paint grid/stones/last-move/win-line; emit
   `cellClicked(Position)` from mouse events (click → nearest intersection).
2. `GameController` state machine + worker thread (local board copy → search → queued
   signal back); show "thinking" state.
3. `MainWindow`: board + toolbar/menu (New, Undo, Resign), difficulty selector, status bar.
4. `main.cpp` wiring.

**Acceptance:** `nix run .` launches; a full game is playable to win/draw; UI stays
responsive during the AI turn; win line and last move are shown.

---

## Phase 5 — Polish & hardening

Tasks:
1. Enable `-Wall -Wextra` (and `-Werror` in checks); fix all warnings.
2. Add the remaining niceties from FR10–FR12 as time permits (status messages, etc.).
3. Review code for refactor/simplification; update docs if contracts changed.
4. Verify `nix flake check` and `nix build` from a clean clone.

**Acceptance:** all acceptance criteria in `REQUIREMENTS.md` pass.

---

## Testing discipline (applies to every phase)
- Write the failing test first; confirm it **fails as expected** before implementing.
- Cover edges and failure modes, not just happy paths (board edges, overline, full
  board, time budget).
- Core is Qt-free and hermetic — no filesystem, no network, no Qt in tests.

## Risks & mitigations
| Risk | Mitigation |
|------|-----------|
| Qt platform plugin not found at runtime | `qt6.wrapQtAppsHook` wraps the binary so plugins resolve; verify with `nix run`. |
| AI too slow / hangs the UI | Time-budgeted iterative deepening + background thread + candidate pruning. |
| Alpha-beta search subtle bugs (e.g., overline, off-by-one on diagonals) | Exhaustive unit tests on `WinDetector` and targeted engine tests. |
| nixpkgs Qt 6 API drift across branches | Pin nixpkgs; keep `find_package(Qt6 COMPONENTS Widgets)` minimal. |
| Over-engineering the AI | Start with depth 2–4 + pattern eval; only add TT/Zobrist if profiling demands it. |

## Definition of done
- [ ] `nix develop` + `cmake --build build` + `ctest` all green, zero warnings.
- [ ] `nix build .#` and `nix run .#` work; `nix flake check` passes.
- [ ] A human can play a full game vs the AI offline and reach a win/draw.
- [ ] `docs/REQUIREMENTS.md` and `docs/ARCHITECTURE.md` reflect the shipped behavior.
