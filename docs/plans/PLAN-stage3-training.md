# Plan — Stage 3: self-play data generation + Python training

Parent plan: [`PLAN-neural-mcts.md`](PLAN-neural-mcts.md) (Stage 3). This is the
detailed execution plan for that stage. Requirements: FR13–FR15, NFR7.

## Goal
Produce a **trained `*.gnn` model** offline (developer tooling only — never in
the shipped binary) such that the Neural MCTS engine plays at least Classic-Hard
strength, then ship that model alongside the game.

Two data sources feed the trainer, both emitted as the **same record format** so
the Python trainer consumes them identically:

1. **Bootstrap (supervised pre-training)** — a headless C++ tool plays
   `AlphaBetaEngine` vs itself and emits games labelled with the alpha-beta move
   (softened) as the policy target and the game outcome as the value target.
   Cheap warm start: alpha-beta is ~100× faster per game than MCTS self-play.
2. **Self-play (RL)** — a headless C++ tool plays `MctsEngine` (PUCT + root
   Dirichlet noise) vs itself and emits games labelled with the root **visit-count
   distribution** as the policy target and the game outcome as the value target.

## The game-record format (authoritative)

Little-endian, versioned, self-describing. This is the contract between the C++
generators and the Python trainer; `src/core/GameRecord.{hpp,cpp}` is the
reference implementation and its tests pin the byte layout.

```
Stream      := StreamHeader  Game*
StreamHeader:= magic[9] "GOMOKUREC"  u32 version (= 1)
Game        := u32 numPositions  PositionRecord[numPositions]
PositionRecord := f32 planes[900]  f32 policy[225]  f32 value
```

- `planes[900]` — the network input, **exactly** `BoardEncoder::encode(board,
  toMove)` output: 4 planes × 15×15, channel-major, each plane row-major
  (`idx = r*15 + c`). Plane 0 = to-move stones, plane 1 = opponent stones,
  plane 2 = constant 1, plane 3 = last-move marker. The player to move is
  therefore recoverable from the planes (plane 2 fill + which colour is on
  plane 0), so the record carries no separate `toMove` field.
- `policy[225]` — a probability distribution over cells (row-major, sums to 1).
  Self-play: normalized root visit counts. Bootstrap: label-smoothed one-hot
  over the chosen alpha-beta move (see below).
- `value` — the game outcome from the perspective of the player to move at
  **this** position: `+1` win, `-1` loss, `0` draw.

Per-position size = 900 + 225 + 1 = 1126 f32 = 4504 bytes.

### Bootstrap policy target (label smoothing)
The alpha-beta move is a brittle hard label; the plan requires it be "softened
so the policy head is not brittle". For a position with legal cells `L`
(empty cells) and chosen move `m`:

```
target[c] = (1 - eps) * [c == m]  +  eps / |L|   for c in L
target[c] = 0                                    for c not in L
```

`eps = 0.1` (constant, tunable). Sum is exactly 1; illegal cells carry no mass.

## Increments

### Increment 1 — C++ record-generation layer (this pass)
Deliverables:
- `src/core/GameRecord.{hpp,cpp}` (Qt-free, hermetic):
  - `PositionRecord` / `GameRecord` structs.
  - `encodeStream` / `decodeStream` over `std::vector<std::uint8_t>` with full
    validation (bad magic, bad version, truncation, inconsistent `numPositions`).
  - `labelSmoothedPolicy(moveIdx, legalMask, eps)` pure helper.
- `src/core/SelfPlay.{hpp,cpp}` (Qt-free): the game orchestration.
  - `generateBootstrapGame(seed, params) -> GameRecord` — AlphaBeta vs
    AlphaBeta, value = outcome from each position's to-move perspective.
  - `generateSelfPlayGame(seed, params) -> GameRecord` — MCTS vs MCTS via the
    new `MctsEngine::selfPlay`, value = outcome. Pre: a model is loaded.
- `MctsEngine::selfPlay(board, player, params) -> SelfPlayResult` (with
  `{ Move move; std::array<float,225> policy; }`): PUCT search with **root
  Dirichlet noise** (`alpha = 0.3`, noise weight `eps = 0.25`), driven by
  `params.seed` (seed 0 treated as a fixed default so it is still
  reproducible). Returns the argmax-visit-count move (row-major tie-break) and
  the normalized visit-count distribution. The existing deterministic
  `findBestMove` is unchanged (its tests stay green).
- Headless tools (thin `main()` wrappers over the core generators, pure C++,
  **not installed**, developer tooling only):
  - `src/tools/bootstrap.cpp` → `gomoku-bootstrap --games N --depth D
    --time-ms T --seed S --out FILE`.
  - `src/tools/self_play.cpp` → `gomoku-selfplay --model FILE.gnn --games N
    --sims M --time-ms T --seed S --out FILE`.
  - CMake: `add_executable`, link `gomoku_core`, no `install`, not in the
    shipped app. (They still build under `nix flake check`.)
- Tests (TDD, hermetic, memory buffers only):
  - `test_game_record.cpp`: round-trip; malformed rejection; label smoothing
    sums to 1 / puts mass on the chosen move / zero on illegal cells.
  - `test_selfplay.cpp`: bootstrap game is legal & deterministic for a seed;
    self-play game legal & deterministic; policy distributions sum to 1;
    record → bytes → decode round-trips; every generated game is replayed
    from its planes and each position's value matches the replayed winner
    (+1/-1/0) from that position's to-move perspective.
- Update `docs/ARCHITECTURE.md` (new core components + tools) and the parent
  plan's definition-of-done checkboxes where applicable.

### Increment 2 — Python model + `.gnn` exporter (round-trip gate)
**Status: complete.** `training/model.py`, `training/export_gnn.py`,
`training/gnn_format.py`, `gomoku-dump-weights`/`gomoku-nn-eval` tools, and the
stdlib round-trip + torch forward gates are in (see `PLAN-stage3-inc2.md`).
- `training/model.py`: PyTorch model matching the Stage-2 architecture exactly
  (conv/Bn/ReLU shapes as in `src/core/NeuralNet.cpp`; `track_running_stats=True`,
  `eps=1e-5`).
- `training/export_gnn.py`: writes `*.gnn` in the **exact** tensor order of
  `src/core/Weights.cpp` (little-endian f32, channels-first). `Weights.cpp` is
  the single source of truth.
- A **round-trip gate**: a stdlib-only Python test writes a `.gnn` with known
  nonzero values at known offsets and checks the bytes against the C++ loader;
  plus a C++↔PyTorch forward-pass tolerance comparison (rel 1e-4), run as a
  developer script (not `ctest` — needs PyTorch).

### Increment 3 — Python training loop + head-to-head gate
**Status: complete.** `training/game_record.py` (Python GameRecord decoder),
`training/train.py` (SL→RL loop, `hardware.select_device()`, periodic `.gnn`
export), `src/tools/headtohead.cpp` (`gomoku-headtohead`), and the
`test_game_record.py`/`test_train.py`/`test_headtohead.py` gates are in (see
`PLAN-stage3-inc3.md`).
- `training/train.py`: SL (cross-entropy on softened policy + MSE on value) then
  RL (cross-entropy on visit counts + MSE on value, self-play every N steps),
  periodic `.gnn` export.
- Head-to-head gate vs the classic engine (developer script, not `ctest`).

### Increment 4 — ship a trained model + end-to-end verification
**Status: complete.**
- Drop the best model into the game's data path; verify `MctsEngine` loads it
  and the Neural engine plays a full offline game at ≥ Classic-Hard strength.

## Acceptance criteria
- All increments: `nix develop -c ctest --test-dir build` green, `-Werror`
  clean, `nix flake check` passes; new files registered with `git add -N`.
- Increment 1: `gomoku-bootstrap` and `gomoku-selfplay` each produce a stream
  that `GameRecord::decodeStream` round-trips; unit tests are hermetic.
- Increment 2: `Weights::parse(export_gnn.py output)` round-trips and the
  C++/PyTorch forward passes agree within rel 1e-4.
- Increment 4: definition of done in `PLAN-neural-mcts.md`.

## Risks
| Risk | Mitigation |
|------|-----------|
| Record format drift between C++ and Python | Format pinned by `GameRecord` tests + the Python round-trip gate; `Weights.cpp` is the .gnn source of truth. |
| Bootstrap games all identical (AlphaBeta is deterministic) | A distinct seed per game (seed = gameIndex + 1) drives AlphaBeta's root-candidate shuffle → variety. |
| Dirichlet sampler not in stdlib | Use `std::gamma_distribution`; Dirichlet = normalize independent Gamma(alpha,1) draws. |
| Plane-only records are large (~4.5 KB/position) | Fine for offline dev tooling; a compact move-list variant is a possible later optimization, not required. |
| Self-play needs a model before any model is trained | Bootstrap (SL) produces the first model; self-play only runs after SL. Documented ordering. |
