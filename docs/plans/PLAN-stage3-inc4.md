# Plan — Stage 3, Increment 4: ship a trained model + end-to-end verification

Parent: [`PLAN-stage3-training.md`](PLAN-stage3-training.md). Requirements
FR13–FR15, NFR7. Increments 1 (C++ record generation + tools), 2 (Python model +
`.gnn` exporter + round-trip gate), and 3 (Python training loop + head-to-head
gate) are **complete**. This increment produces a real trained model, verifies it
beats Classic-Hard alpha-beta, and ships it with the game so the Neural engine
works out of the box (FR14).

## Goal
1. Run a **real** training run (SL bootstrap then RL self-play, larger than the
   smoke defaults) producing a model that wins ≥ 50% of decided games against
   Classic-Hard alpha-beta (depth 6 / 2000 ms) via `test_headtohead.py`.
2. **Ship** the best model so the Neural engine loads it by default — no
   `GOMOKU_MODEL_PATH` needed to use the Neural engine.
3. Tick the definition-of-done boxes and update `REQUIREMENTS.md` /
   `ARCHITECTURE.md`.

## Current state (verified at `011ebdd`)
All Inc 1–3 gates green: `ctest` 75/75, `nix flake check`,
`test_hardware`/`test_gnn_roundtrip`/`test_forward`/`test_game_record`/
`test_train`/`test_headtohead` (report-only). Machine is CPU-only
(torch 2.13.0+cpu, 16 cores), so training is the throughput gate.

Measured costs on this machine (single-threaded C++ forward ≈ 2.4 ms):
- `gomoku-bootstrap`: depth 3 ≈ 6.4 s/game, depth 4 (time-ms 2000) ≈ 21 s/game.
- `gomoku-selfplay`: ≈ 2.4 ms × sims × positions-per-game (≈46). 100 sims ≈ 11 s/game,
  400 sims ≈ 44 s/game, 800 sims ≈ 88 s/game.
- Head-to-head at 1600 sims / 2000 ms: the 2000 ms deadline caps MCTS at ≈ 800
  effective sims, so ≈ 2 s/MCTS move.

## Contracts

### Model shipping (new)
- The trained model is committed at `resources/model.gnn` (~720 KB for N=4/C=16).
- **flake.nix source filter** currently drops every `*.gnn`; it must whitelist the
  exact path `resources/model.gnn` so `nix build`/`nix flake check` include it.
- **CMake** installs it: `install(FILES resources/model.gnn DESTINATION
  ${CMAKE_INSTALL_DATADIR}/gomoku)` → the store layout is `bin/gomoku` +
  `share/gomoku/model.gnn`.
- **MainWindow** model lookup order (in `MainWindow::loadModel`, replacing
  `loadModelFromEnv`):
  1. `GOMOKU_MODEL_PATH` (explicit override; wins, never silently falls back),
  2. `<applicationDirPath>/model.gnn` (dev: model next to the binary),
  3. `<applicationDirPath>/../share/gomoku/model.gnn` (installed bundle).
  A successful load logs the resolved source to stderr (`qInfo`); a set-but-bad
  env path logs a warning.

### `train.py` parallel generation (new)
Self-play and bootstrap are single-threaded C++ tools whose games are independent.
Add `--jobs J` (default 1) so each generation phase runs `J` tool processes in
parallel (distinct per-game seeds, balanced chunks) and byte-concatenates the
resulting `.rec` streams (strip each 13-byte header, prepend one). `J=1` keeps the
exact existing single-process behavior. Two new pure helpers, tested hermetically:

- `balanced_offsets(total, jobs) -> list[int]` — balanced chunk start offsets.
- `merge_rec_files(paths, out) -> None` — byte-level `.rec` merge (no full decode).

`game_record.py` stays a stdlib decode+round-trip module; `gnn_format.py` stays
encode-only (the model is never reloaded into PyTorch).

## Deliverables
1. `resources/model.gnn` — the trained, verified model.
2. `flake.nix` — whitelist `resources/model.gnn` in the source filter.
3. `CMakeLists.txt` — install the model to `share/gomoku/`.
4. `src/ui/MainWindow.{hpp,cpp}` — default-model fallback lookup.
5. `training/train.py` — `--jobs` + `balanced_offsets` + `merge_rec_files`.
6. `training/tests/test_train.py` — hermetic tests for the two new helpers.
7. Docs: tick Increment 4 in `PLAN-stage3-training.md`, tick the
   definition-of-done in `PLAN-neural-mcts.md`, update `REQUIREMENTS.md` +
   `ARCHITECTURE.md`, and note the shipping gotchas in `AGENTS.md`.

## Training budget (targets, tuned to fit ~1–2 h of wall-clock)
- SL: `gomoku-bootstrap --games 240 --depth 4 --time-ms 2000`, `--jobs 8`.
- RL: 4–6 iterations of `gomoku-selfplay --games 200 --sims 400`, `--jobs 8`.
- Save `model.sl.gnn` + `model.rl<N>.gnn`; keep the best by the head-to-head gate.

## Verification
```bash
nix develop -c cmake -S . -B build -G Ninja && nix develop -c cmake --build build
nix develop -c ctest --test-dir build                          # 75 stay green
cd training
uv run python tests/test_train.py                              # new helper tests
uv run python tests/test_headtohead.py --tool ../build/gomoku-headtohead \
    --model ../resources/model.gnn --games 20 --min-winrate 0.5   # strength gate
nix flake check
nix build && test -f result/share/gomoku/model.gnn             # bundling
QT_QPA_PLATFORM=offscreen result/bin/gomoku                    # smoke: loads bundled model
```

## Acceptance criteria
- A real SL→RL run completes and exports `model.sl.gnn` + ≥1 `model.rl<N>.gnn`.
- `test_headtohead.py --min-winrate 0.5` passes against `resources/model.gnn`.
- The shipped game loads the bundled model by default (no env var); the Neural
  engine is enabled out of the box.
- All prior gates green: `ctest` (75), `test_hardware`, `test_gnn_roundtrip`,
  `test_forward`, `test_game_record`, `test_train`, `test_headtohead`,
  `nix flake check`; `-Werror` clean.
- New files `git add -N`; `git diff --cached` empty (the sole documented
  exception — `training/uv.lock` untracked — remains so).

## Result (what actually shipped)
- Shipped `resources/model.gnn` = the `model.rl3.gnn` checkpoint of the
  `inc4-training3` run (SL: 512 games, depth 6 / 2000 ms, 6 epochs; RL: 5 iters
  of 160-game / 800-sim self-play with the SL anchor and 16-way parallel
  generation).
- **Head-to-head vs Classic-Hard (depth 6 / 2000 ms, MCTS 1600 sims ≈ 800
  effective under the 2 s budget): 50% win rate (10–10 over 20 games), reproduced
  across 5 independent 20-game runs.** The neural engine wins every game as
  Black (first mover) and loses every game as White — genuine parity with
  Classic-Hard (freestyle gomoku's first-mover advantage is decisive, and the
  small net cannot overcome it as White).
- Key findings that shaped the run: (1) RL with no SL anchor catastrophically
  forgets the teacher (0% win rate after 4 iters); (2) 400-sim self-play is too
  weak to exceed the teacher — 800 sims (≈ eval strength) is required;
  (3) the model peaks at RL iteration 3 and degrades past it (both independent
  runs' `rl3` checkpoint is the best).
- `MctsEngine::loadModel` loads the shipped bytes; the game finds the bundled
  model via `<applicationDirPath>/../share/gomoku/model.gnn` (verified with an
  offscreen smoke launch: "Neural engine: loaded bundled model from …").
