# Gomoku — Architecture

## Design principles
1. **Engine / UI split.** Every rule and every AI decision lives in a Qt-free C++
   core library. The Qt layer is a thin presentation + orchestration shell. This is
   what makes the AI unit-testable with a plain test runner.
2. **Stateless search.** The search engine takes the board by value (or copy) and
   returns a move. No shared mutable state → trivially safe to run on a background
   thread without locks.
3. **Contracts.** Each non-trivial function documents pre/post-conditions and
   invariants; illegal states are rejected with assertions ("fail fast").

## Layers

```
┌───────────────────────────────────────────────────────────┐
│  UI layer (Qt 6 Widgets)                                  │
│  MainWindow ── BoardWidget (QPainter custom widget)       │
│       │                                                   │
│  GameController (QObject, owns state machine + AI thread) │
└───────────────┬───────────────────────────────────────────┘
                │  plain C++ calls (no Qt types)
┌───────────────▼───────────────────────────────────────────┐
│  Core engine (Qt-free, header-only-friendly static lib)   │
│  Board · WinDetector · Evaluator · PatternTable           │
│  SearchEngine (facade) → AlphaBetaEngine | MctsEngine     │
│  NeuralNet · BoardEncoder · Weights (Neural engine)       │
└───────────────────────────────────────────────────────────┘
```

The dependency arrow points **upward only**: core never includes Qt, UI never
implements game logic. This boundary is enforced by the CMake target graph
(`gomoku` links `gomoku_core`; `gomoku_core` has no Qt dependency).

## Core engine (`src/core/`)

| Component | Responsibility |
|-----------|----------------|
| `types.hpp` | `Player { None, Black, White }`, `Position`, `Move`, board constants (`kSize = 15`). |
| `Board` | Fixed 15×15 grid; `place()`, `undo()`, `isEmpty()`, `isFull()`, `inBounds()`, move history for undo. |
| `WinDetector` | Pure function: `winnerOf(board, lastMove) → Player`. Scans the 4 directions through the last stone, counting consecutive same-color stones. `findWinningLine()` returns the 5 cells for UI highlighting. |
| `PatternTable` | Maps local stone patterns (e.g. `_XXXX_`, `_XXX_`, `_OOO_`) to scores, built once at startup. |
| `Evaluator` | Static board score = sum of pattern scores for Black minus White, scanned in all 4 directions. |
| `SearchEngine` | **Facade**: `findBestMove(board, player, params) → Move`, dispatching on `params.engine`. `difficulty(level)` maps to depth/time (Classic) and simulation count (Neural). |
| `AlphaBetaEngine` | The original search: negamax with alpha-beta pruning, iterative deepening, candidate-move generation (only cells within Chebyshev distance 2 of an existing stone), and an optional time budget. |
| `MctsEngine` | Monte-Carlo tree search (PUCT) guided by a trained policy/value network. **Requires a loaded model**; throws if none is loaded. |
| `NeuralNet` | Forward-only convolutional residual network (policy + value heads), pure C++ float32, no external deps. |
| `BoardEncoder` | Maps a `Board` to the input tensor planes (own/opponent stones, to-move, last-move marker). |
| `Weights` | Loads the `*.gnn` weight file (magic + version + layout + raw f32 tensors). |

### Search engines (switchable)
`SearchEngine::findBestMove` is a **facade** that selects an engine via
`SearchParams::engine` (`EngineKind { AlphaBeta, Mcts }`).

**Classic (`AlphaBetaEngine`)** — the original deterministic search:
- **Negamax** (single function, no separate min/max cases) with **alpha-beta pruning**.
- **Iterative deepening**: try depth 1, 2, … up to the cap or until the time budget
  is exhausted; always keep the best move from the last completed depth.
- **Candidate generation** prunes the branching factor from 225 to a few dozen by
  considering only cells adjacent to existing stones.
- **Evaluation** is the pattern-based heuristic (`PatternTable` + `Evaluator`),
  applied at leaf nodes; a terminal win/loss gets a large ±score with depth
  adjustment (prefer faster wins / slower losses).
- Move ordering (winning moves first, then near previous move) improves pruning.

**Neural (`MctsEngine`)** — AlphaZero-style, requires a trained model:
- **PUCT** tree search over the legal moves, with a simulation budget
  (`SearchParams::mctsSimulations`).
- Each leaf is evaluated by `NeuralNet` (policy prior over 225 moves + value in
  [−1, 1]); the board is encoded by `BoardEncoder`.
- Deterministic (no root noise and no RNG in play); ties break in row-major order.
  Dirichlet noise is deferred to Stage 3 self-play.
- **No heuristic fallback**: `MctsEngine::findBestMove` throws if no model is loaded
  (`isModelAvailable()` is false), and the UI disables the Neural option.

### Contracts (examples)
- `Board::place(Move m)`: **pre** `inBounds(m)` and `isEmpty(m)` and no current winner;
  **post** `cell(m) == m.player`, history length +1.
- `WinDetector::winnerOf(b, last)`: **pre** `last` is a legal, occupied cell;
  **post** returns `None` unless exactly the line through `last` completes 5.
- `SearchEngine::findBestMove(b, p, params)`: **pre** `b` has ≥1 empty cell and no winner;
  **post** returned move is legal and empty.

## UI layer (`src/ui/`)

| Component | Responsibility |
|-----------|----------------|
| `BoardWidget` | Custom `QWidget`; paints grid, stones, last-move marker, and winning line via `QPainter`; converts mouse clicks to board positions and emits `cellClicked(pos)`. |
| `GameController` | `QObject` state machine (WaitingForPlayer / AiThinking / GameOver). Receives clicks, applies moves to a `Board`, triggers win/draw checks, spawns the AI search on a `std::jthread` (or `QThread`), and emits UI-update signals. |
| `MainWindow` | Composes `BoardWidget` + menu/toolbar (New / Undo / Resign, difficulty selector) + status bar. |

**Threading model:** when it is the AI's turn, `GameController` runs
`SearchEngine::findBestMove` on a worker thread with a local copy of the board.
On completion it posts the result back via a queued signal; the UI thread then
applies the move. Because the search is stateless, no locks are needed. A "thinking"
indicator is shown while the worker runs.

## Build (`CMakeLists.txt`)
- `gomoku_core` — `STATIC` library from `src/core`, C++23, `-Wall -Wextra`.
- `gomoku` — Qt Widgets executable linking `gomoku_core`; `find_package(Qt6 COMPONENTS Widgets)`.
- `gomoku_tests` — Catch2 test executable linking `gomoku_core`; registered with CTest.
- `CMAKE_EXPORT_COMPILE_COMMANDS=ON` for editor/clangd integration.

## Testing strategy
- **Unit tests** (Catch2) cover the entire core: board placement/undo/full/draw,
  win detection in all 4 directions and both diagonals, overline, evaluator
  symmetry (`score(b, Black) == -score(swapped b, White)`), and engine behaviors
  (finds immediate win, blocks open four, respects time budget, deterministic given
  a seed).
- **Hermetic**: no tests touch Qt, the filesystem, or the network.
- The UI is kept intentionally thin and is verified manually (and via smoke launch
  in `nix run`), not by automated widget tests in the first pass.

## Project layout
```
gomoku/
├── flake.nix
├── CMakeLists.txt
├── .gitignore
├── docs/
│   ├── REQUIREMENTS.md
│   └── ARCHITECTURE.md
├── src/
│   ├── core/
│   │   ├── types.hpp
│   │   ├── Board.hpp / Board.cpp
│   │   ├── WinDetector.hpp / WinDetector.cpp
│   │   ├── PatternTable.hpp / PatternTable.cpp
│   │   ├── Evaluator.hpp / Evaluator.cpp
│   │   ├── SearchEngine.hpp / SearchEngine.cpp   (facade + difficulty)
│   │   ├── AlphaBetaEngine.hpp / AlphaBetaEngine.cpp
│   │   ├── MctsEngine.hpp / MctsEngine.cpp
│   │   ├── NeuralNet.hpp / NeuralNet.cpp         (Stage 2)
│   │   ├── BoardEncoder.hpp / BoardEncoder.cpp   (Stage 2)
│   │   └── Weights.hpp / Weights.cpp             (Stage 2)
│   ├── ui/
│   │   ├── MainWindow.hpp / MainWindow.cpp
│   │   ├── BoardWidget.hpp / BoardWidget.cpp
│   │   └── GameController.hpp / GameController.cpp
│   └── main.cpp
└── tests/
    ├── test_board.cpp
    ├── test_windetector.cpp
    ├── test_evaluator.cpp
    └── test_engine.cpp
```
