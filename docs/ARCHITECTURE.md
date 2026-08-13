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
| `MctsEngine` | Monte-Carlo tree search (PUCT) guided by a trained policy/value network. **Requires a loaded model**; throws if none is loaded. `findBestMove` is deterministic; `selfPlay` adds root Dirichlet noise and returns the root visit-count policy target for training. |
| `NeuralNet` | Forward-only convolutional residual network (policy + value heads), pure C++ float32, no external deps. |
| `BoardEncoder` | Maps a `Board` to the input tensor planes (own/opponent stones, to-move, last-move marker). |
| `Weights` | Loads the `*.gnn` weight file (magic + version + layout + raw f32 tensors). |
| `GameRecord` | Training-record structs (`PositionRecord` = planes + policy + value, `GameRecord` = ordered positions) and the binary `encodeStream`/`decodeStream` plus `labelSmoothedPolicy`. The byte format is the contract shared with the Python trainer (see `docs/plans/PLAN-stage3-training.md`). |
| `SelfPlay` | Headless game generation: `generateBootstrapGame` (AlphaBeta vs AlphaBeta, label-smoothed policy) and `generateSelfPlayGame` (MCTS vs MCTS via `MctsEngine::selfPlay`, visit-count policy) emit `GameRecord`s for supervised and RL training. |

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
- Deterministic `findBestMove` (no root noise); `selfPlay` mixes root Dirichlet
  noise into the priors and returns the visit-count policy target for training.
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
- `gomoku-bootstrap`, `gomoku-selfplay` — headless training-data tools (Stage 3,
  developer tooling only; link `gomoku_core`, never installed).
- `gomoku-dump-weights`, `gomoku-nn-eval` — Stage 3 round-trip-gate tools
  (developer tooling only; link `gomoku_core`, never installed). The first
  parses a `.gnn` and dumps every tensor in file order; the second runs
  `NeuralNet::evaluate` on 900 input planes and prints logits + value.
- `gomoku-headtohead` — Stage 3 head-to-head gate tool (developer tooling
  only; link `gomoku_core`, never installed). Plays the neural (MCTS) engine
  against the classic (alpha-beta) engine at Classic-Hard strength and prints
  the win/loss/draw score.
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

## Training toolchain (Python, `training/`)

Python is allowed **for training only** — never in the shipped binary. The
PyTorch side is a `uv`-managed project (`training/pyproject.toml`, venv lives in
`training/.venv`, git-ignored and excluded from the Nix source filter).

| File | Responsibility |
|------|----------------|
| `gnn_format.py` | **Stdlib-only** single Python source of truth for the `.gnn` byte layout: magic/version/header, ordered tensor spec (mirroring `Weights.cpp`), `serialize`/`pack_f32s`/`tensor_offsets`. No torch import. |
| `model.py` | `GomokuNet` (default `num_blocks=4`, `channels=16`): input conv+BN+ReLU, `num_blocks` residual blocks, policy head (conv2 → flatten 450 → linear → 225 logits), value head (conv1 → flatten 225 → linear 256 → ReLU → linear 1 → tanh). BatchNorm `eps=1e-5`, `track_running_stats=True`. |
| `export_gnn.py` | `export_gnn(model) -> bytes` + `--out` CLI. Reads tensors by **explicit attribute access** in the exact `Weights.cpp` order and writes little-endian f32 `.gnn`. |
| `hardware.py` | **Stdlib-only** hardware-aware torch selection: `detect_gpu()` (via `nvidia-smi`), `torch_index_url()` (CPU vs the highest compatible CUDA index), `select_device()` (runtime `cuda`/`cpu`). No torch import at module top — the sync script uses it before torch is installed. |
| `scripts/sync.sh` | **The one install entry point**: detects the machine (CPU vs NVIDIA) and runs `uv sync --index <torch-index> --index-strategy first-index`. CPU boxes get the small CPU wheel; GPU boxes get a compatible CUDA wheel. `uv.lock` is **untracked** (it encodes one hardware's torch and differs per machine). |
| `game_record.py` | **Stdlib-only** Python decoder for the GameRecord binary stream (mirrors `src/core/GameRecord.cpp`): magic `GOMOKUREC`, u32 version=1, then per game `u32 numPositions` + per position `f32 planes[900] + f32 policy[225] + f32 value`. `decode_stream`/`decode_file`; `encode_stream` exists only for the round-trip gate. |
| `train.py` | **SL→RL training loop**: bootstrap records (label-smoothed policy CE + value MSE), export `.gnn`, then repeated `gomoku-selfplay --model <gnn>` + RL training (visit-count policy CE + value MSE), with periodic `.gnn` export. Uses `hardware.select_device()` for the runtime device; the model is kept in memory (never reloaded from `.gnn`). |
| `tests/test_hardware.py` | **Hermetic** tests for `hardware.py`: injected `GPUInfo` + fake `nvidia-smi` runs (no GPU/network/torch needed for the mapping logic). |
| `tests/test_gnn_roundtrip.py` | **Stdlib-only round-trip gate**: writes a `.gnn` with known values at known offsets (via `gnn_format`), checks the raw bytes, then runs `gomoku-dump-weights` and asserts the C++ loader reports identical values. |
| `tests/test_forward.py` | **C++↔PyTorch forward gate** (needs torch): random `GomokuNet`, export → `gomoku-nn-eval`, compare all 225 logits + value within rel 1e-4 (abs floor 1e-5). Stays on CPU (it compares against the CPU C++ engine). Developer script, not `ctest`. |
| `tests/test_game_record.py` | **GameRecord decoder gate** (stdlib-only): hermetic decode of struct-built bytes + malformed rejection, `encode∘decode` round-trip, and integration against real `gomoku-bootstrap` output (well-formedness). |
| `tests/test_train.py` | **Hermetic torch test** of the pure `train.py` helpers: `records_to_tensors` shapes, distribution-target cross-entropy, and that an optimizer step changes a parameter. No C++ tools. |
| `tests/test_headtohead.py` | **Head-to-head gate driver**: runs `gomoku-headtohead` and reports the neural engine's win rate vs Classic-Hard alpha-beta; report-only unless `--min-winrate` is set (Increment 4 turns on the threshold). |

Gate commands (from `training/`):

```bash
scripts/sync.sh                                      # hardware-aware torch install
uv run python tests/test_hardware.py                 # hermetic, no torch needed
uv run python tests/test_index_resolve.py            # network-gated: every torch index resolves
uv run python tests/test_gnn_roundtrip.py --tool ../build/gomoku-dump-weights  # stdlib only
uv run python tests/test_forward.py --tool ../build/gomoku-nn-eval        # needs torch
uv run python tests/test_game_record.py --tool ../build/gomoku-bootstrap  # decoder gate
uv run python tests/test_train.py                     # hermetic torch helpers
uv run python tests/test_headtohead.py --tool ../build/gomoku-headtohead \
    --model <model.gnn> --games N                     # head-to-head vs classic engine
```

Training run (SL then RL):

```bash
uv run python train.py \
    --bootstrap-tool ../build/gomoku-bootstrap \
    --selfplay-tool ../build/gomoku-selfplay \
    --outdir .pi/training --sl-games 200 --rl-iters 3
```

Runtime device selection: training code should use `hardware.select_device()`
(`cuda` when `torch.cuda.is_available()`, else `cpu`; it warns if a GPU is
present but the installed torch build has no CUDA).

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
│   │   ├── Weights.hpp / Weights.cpp             (Stage 2)
│   │   ├── GameRecord.hpp / GameRecord.cpp       (Stage 3)
│   │   └── SelfPlay.hpp / SelfPlay.cpp           (Stage 3)
│   ├── tools/
│   │   ├── bootstrap.cpp                         (Stage 3: alpha-beta self-play → records)
│   │   ├── self_play.cpp                         (Stage 3: MCTS self-play → records)
│   │   ├── dump_weights.cpp                      (Stage 3: .gnn → tensor dump, round-trip gate)
│   │   ├── nn_eval.cpp                           (Stage 3: .gnn + planes → forward pass)
│   │   └── headtohead.cpp                        (Stage 3: neural vs classic engine gate)
│   ├── ui/
│   │   ├── MainWindow.hpp / MainWindow.cpp
│   │   ├── BoardWidget.hpp / BoardWidget.cpp
│   │   └── GameController.hpp / GameController.cpp
│   └── main.cpp
├── training/                                    (Stage 3: Python training toolchain, uv-managed)
│   ├── pyproject.toml
│   ├── gnn_format.py                            (.gnn layout source of truth, stdlib-only)
│   ├── model.py                                 (PyTorch GomokuNet)
│   ├── export_gnn.py                            (.gnn exporter)
│   ├── game_record.py                           (GameRecord stream decoder, stdlib-only)
│   ├── train.py                                 (SL→RL training loop)
│   └── tests/
│       ├── test_gnn_roundtrip.py                (stdlib round-trip gate)
│       ├── test_forward.py                      (torch forward gate)
│       ├── test_game_record.py                  (GameRecord decoder gate)
│       ├── test_train.py                        (hermetic train.py helpers)
│       └── test_headtohead.py                   (neural-vs-classic gate driver)
└── tests/
    ├── test_board.cpp
    ├── test_windetector.cpp
    ├── test_evaluator.cpp
    ├── test_engine.cpp
    ├── test_engine_dispatch.cpp
    ├── test_neural.cpp
    ├── test_mcts.cpp
    ├── test_game_record.cpp      (Stage 3)
    └── test_selfplay.cpp         (Stage 3)
```
