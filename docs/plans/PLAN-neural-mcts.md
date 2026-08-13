# Plan — Neural MCTS engine (switchable with the classic alpha-beta AI)

Requirements: [`docs/REQUIREMENTS.md`](../REQUIREMENTS.md) (FR13–FR15, NFR7).
Architecture: [`docs/ARCHITECTURE.md`](../ARCHITECTURE.md).

## Goals
Add a second AI engine — **MCTS guided by a trained policy/value network**
(AlphaZero-style) — alongside the existing **pattern-heuristic alpha-beta** engine.
The player switches between them at runtime. The shipped game stays pure C++
(offline, no runtime network); only the *training* toolchain may use Python.

## Decisions (confirmed with user)
1. **Both engines coexist, switchable at runtime**, behind the same
   `SearchEngine::findBestMove(board, player, params) → Move` facade.
2. **Python (PyTorch) is allowed for training only**, never in the shipped binary.
3. **MCTS requires trained weights** — there is **no** pattern-evaluator fallback.
   With no model loaded, the Neural engine is unavailable (UI item disabled, and
   `MctsEngine::findBestMove` throws as defense-in-depth).
4. Engine selection is orthogonal to difficulty: `EngineKind` is a field of
   `SearchParams`; difficulty presets fill the strength knob for *both* engines
   (`maxDepth`/`timeBudgetMs` for Classic, `mctsSimulations` for Neural).

## Network spec (tunable in Stage 2/3)
- **Input**: 4 binary planes of 15×15 — current player's stones, opponent's stones,
  a constant "player to move" fill, and the last-move marker.
- **Input projection**: `conv3×3(4 → C) → BN → ReLU`.
- **Body**: `N` residual blocks (`N` ≈ 4–8), each
  `conv3×3(C) → BN → ReLU → conv3×3(C) → BN → add-input → ReLU`, `C` ≈ 64–128.
- **Policy head**: `conv3×3(2) → flatten → linear(2·225 → 225)` logits over cells.
- **Value head**: `conv3×3(1) → flatten → linear(225 → 256) → ReLU → linear(256 → 1) → tanh`.
- **Weight file** (`*.gnn`): magic `"GOMOKUNET"` + `u32` version + layout
  (`u32 numBlocks`, `u32 channels`) + raw little-endian f32 tensors in a fixed order.
  One loader in C++; the Python trainer writes the same format.
- **BatchNorm** tensors are eval-mode running stats: running_mean/running_var
  (`track_running_stats=True`), eps = 1e-5.

## Stages (each independently shippable + tested)

### Stage 1 — Engine abstraction, facade, UI switch (this pass)
- Add `EngineKind { AlphaBeta, Mcts }` and extend `SearchParams` with
  `engine` + `mctsSimulations`.
- Move the existing negamax/alpha-beta logic into `AlphaBetaEngine`
  (no behavior change; the existing `tests/test_engine.cpp` stays green).
- Make `SearchEngine` a facade that dispatches on `params.engine`.
- Add `MctsEngine` with `isModelAvailable()` (false for now) and a
  `findBestMove` that throws until weights land.
- UI: `Engine:` combo (Classic / Neural); Neural disabled when no model.
- Tests: facade dispatch, MCTS-without-weights throws, difficulty budgets.

### Stage 2 — Network inference + MCTS
- `BoardEncoder` (board → 4 planes), `NeuralNet` (forward-only conv/residual,
  float32, no external deps), `Weights` (load the `*.gnn` format).
- `MctsEngine`: PUCT tree search; leaf value/policy from the network;
  simulation + time budget; `isModelAvailable()` wired to a loaded model.
  (Root Dirichlet noise is a self-play concern and lands in Stage 3.)
- Tests: encoder correctness, NN forward on a hand-built weight set, MCTS with an
  injected trivial network (finds immediate win, blocks open four, respects budget).

### Stage 3 — Self-play + Python training
- Headless `self_play` C++ tool: plays MCTS-vs-MCTS, emits game records
  (board planes + policy targets + outcome) as a plain binary/text stream.
- Python (PyTorch): training loop (cross-entropy policy, value targets from game
  outcome), weight export to `*.gnn`, and a simple head-to-head gate vs. the
  classic engine. The C++-vs-PyTorch gate is a tolerance comparison (relative
  1e-4), not bit-for-bit: accumulation order and `std::tanh`/`exp` differ by a few
  ULPs.
- Drop a trained model into the game's data path; verify end-to-end.

## Testing discipline
- TDD per stage: write the failing test, confirm it fails, implement, refactor.
- Core stays Qt-free and hermetic; no network/filesystem in unit tests
  (the weight loader is tested against in-memory byte buffers).
- Classic-engine behavior is pinned by the existing engine tests — the refactor
  must not regress them.

## Risks & mitigations
| Risk | Mitigation |
|------|-----------|
| Hand-rolled conv/backprop correctness | Forward-only in C++ (no backprop there); backprop only in PyTorch. NN forward checked against PyTorch outputs on the same weights. |
| CPU-only training is slow | Small net (≈0.5–2M params); self-play is the throughput gate, run it headless/multi-threaded. |
| CPU inference cost | The trained network must stay small (channels ≤ ~32, blocks ≤ ~4) so ~1000 sims/move fits the 2 s budget; if per-expansion `Board` copies dominate, consider lazy board materialization. |
| MCTS nondeterminism breaks reproducibility | Fixed seed → deterministic PUCT tie-breaking; determinism test for a fixed seed. |
| Weights absent at runtime | Neural engine disabled in UI; `MctsEngine::findBestMove` throws (no silent fallback). |

## Definition of done
- [ ] Classic and Neural engines switchable from the UI.
- [ ] Neural disabled (and throws) without a trained model.
- [ ] A trained model plays a full game offline; strength ≥ Classic on Hard.
- [ ] `nix develop -c ctest --test-dir build` green, zero warnings; `nix flake check` passes.
- [ ] `docs/REQUIREMENTS.md` and `docs/ARCHITECTURE.md` reflect the shipped behavior.
