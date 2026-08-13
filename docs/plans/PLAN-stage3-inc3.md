# Plan — Stage 3, Increment 3: Python training loop + head-to-head gate

Parent: [`PLAN-stage3-training.md`](PLAN-stage3-training.md). Requirements FR13–FR15.
Increments 1 (C++ record generation + tools) and 2 (Python model + `.gnn` exporter
+ round-trip gate) are **complete**. This increment adds the Python training loop
that consumes the C++ record streams, and a head-to-head gate that measures the
trained neural engine against the classic alpha-beta engine. It does **not** ship
a model (that is Increment 4).

## Goal
Turn the already-working pieces — `gomoku-bootstrap` / `gomoku-selfplay` (Inc 1),
`GomokuNet` + `export_gnn` (Inc 2) — into a working training loop:

1. **SL (supervised pre-training)** from bootstrap records: policy cross-entropy on
   the label-smoothed alpha-beta move, MSE on the outcome value.
2. **RL (self-play)**: export the SL model, then repeatedly run `gomoku-selfplay
   --model <current.gnn>` and train on the visit-count/outcome records.
3. **Periodic `.gnn` export** after SL and after each RL iteration.
4. A **head-to-head gate** (C++ tool + Python driver) that plays the neural engine
   against the classic engine at Classic-Hard strength and reports win/loss/draw.

## Contracts (authoritative)

### GameRecord stream (Python decoder)
Mirrors `src/core/GameRecord.cpp` exactly. The stream is little-endian:

```
Stream      := magic[9] "GOMOKUREC"  u32 version(=1)  Game*
Game        := u32 numPositions  PositionRecord[numPositions]
PositionRecord := f32 planes[900]  f32 policy[225]  f32 value
```

Per-position = 1126 f32 = 4504 bytes. `planes` are the network input exactly as
`BoardEncoder::encode` emits (4×225, channel-major, each plane row-major).
`policy` sums to 1; `value` ∈ {-1, 0, +1} from the to-move perspective.

### Trainer orchestration (must match this order)
```
gomoku-bootstrap --games N --depth D --time-ms T --seed S --out sl.rec   # SL first
train SL on decode(sl.rec)
export_gnn -> model.gnn
for it in 1..rl_iters:                                                   # RL self-play
    gomoku-selfplay --model model.gnn --games M --sims K --time-ms T --seed S --out rl.rec
    train RL on decode(rl.rec)
    export_gnn -> model.gnn (and a dated copy)
```
Self-play needs a trained model (the C++ tool calls `MctsEngine::loadModel`), so
SL always runs first. The core loop keeps the model **in memory** and only
exports; it never reloads a `.gnn` back into PyTorch (`gnn_format.py` stays
encode-only — do **not** add a `.gnn` decoder).

## Deliverables

### Python (in `training/`, uv-managed, stdlib where noted)
1. `training/game_record.py` — **stdlib-only** decoder for the record stream:
   - `MAGIC = b"GOMOKUREC"`, `VERSION = 1`, `HEADER_SIZE = 13`, `PLANES = 900`,
     `POLICY = 225`, `POSITION_FLOATS = 1126`, `POSITION_BYTES = 4504`.
   - `@dataclass PositionRecord { planes: list[float], policy: list[float], value: float }`.
   - `decode_stream(data: bytes) -> list[list[PositionRecord]]` — raises
     `ValueError` on bad magic, unsupported version, truncation, a `numPositions`
     that overruns the buffer, or trailing bytes. No numpy/torch import.
   - `encode_stream(games) -> bytes` — the inverse (used only by the round-trip
     test; the trainer never encodes). `decode_file(path)` convenience.
   - Use `struct.unpack("<1126f", …)` per position; slice `[0:900]`, `[900:1125]`,
     `[1125]` into planes/policy/value.
2. `training/train.py` — the SL→RL training loop:
   - Pure, import-safe helpers (no top-level side effects; `main()` guarded):
     `flatten_records(games)`, `records_to_tensors(records) -> (planes, policy,
     value)` with shapes `(N,4,15,15)`, `(N,225)`, `(N,1)`; `policy_cross_entropy
     (logits, target)` = `-(target * log_softmax).sum(1).mean()` (target is a
     distribution, not a class index — identical for SL label smoothing and RL
     visit counts); `value_mse(pred, target)`; `train_epoch(...)` mini-batch loop
     (Adam, `policy_loss + value_weight * value_loss`).
   - Device: `hardware.select_device()` (move model + tensors there). The
     C++↔PyTorch forward gate stays on CPU, but the **trainer** may use CUDA.
   - CLI with explicit tool paths (no implicit `build/` lookup):
     `--bootstrap-tool`, `--selfplay-tool`, `--outdir`, `--num-blocks`, `--channels`,
     `--seed`, SL knobs (`--sl-games --sl-depth --sl-time-ms --sl-epochs
     --sl-batch-size --sl-lr`), RL knobs (`--rl-iters --rl-games --rl-sims
     --rl-time-ms --rl-epochs --rl-batch-size --rl-lr`), `--value-weight`,
     `--export-every`. Shells out with `subprocess.run(..., check=True)` and
     surfaces tool stderr on failure.
   - Writes `<outdir>/sl.rec`, `<outdir>/rl.it<N>.rec`, and
     `<outdir>/model.gnn` (always the latest) plus `<outdir>/model.sl.gnn`,
     `<outdir>/model.rl<N>.gnn` copies.
3. `training/tests/test_game_record.py` — decoder gate (run without torch):
   - **Hermetic**: build a stream with `struct` (non-circular, hardcoded bytes),
     assert decode returns the expected values; assert malformed inputs
     (bad magic, bad version, truncated header/position, oversized
     `numPositions`, trailing byte) each raise `ValueError`.
   - **Round-trip**: `encode_stream(decode_stream(x)) == x` for a hand-built
     stream; and decode→encode→decode is idempotent.
   - **Integration**: run `gomoku-bootstrap` (tiny: `--games 3 --depth 1
     --time-ms 2000 --out …`), decode its output, and assert well-formedness:
     every position has 900/225 floats, planes ∈ {0,1}, policy sums to 1 (≈1e-3),
     value ∈ {-1,0,1}, ≥1 game, ≥1 position. Tool path via `--tool`.
4. `training/tests/test_train.py` — hermetic torch test of the pure helpers:
   build `PositionRecord`s in memory (no C++ tools), assert `records_to_tensors`
   shapes/values, and that `policy_cross_entropy`/`value_mse` are finite scalars
   and backprop updates a parameter. Needs torch (developer script, not ctest).
5. `training/tests/test_headtohead.py` — the head-to-head gate driver (needs
   torch only for nothing; it just shells out). Usage
   `--tool ../build/gomoku-headtohead --model <model.gnn> --games N
   [--min-winrate 0.5]`. Parses the tool's stdout (`mcts_wins`, `ab_wins`,
   `draws`, `games`, color splits), prints the win rate, and only fails when
   `--min-winrate` is set and the MCTS win rate over decided games falls short
   (default: report only — Inc 4 turns on the threshold).

### C++ developer tool (`src/tools/headtohead.cpp`, never installed)
`gomoku-headtohead --model FILE.gnn --games N --sims M --depth D --time-ms T
--seed S`. Loads the model (`MctsEngine::loadModel`), then plays N games between
**MCTS (neural)** and **AlphaBeta (classic, Classic-Hard defaults depth 6 / 2000
ms)**, alternating colors each game (game i: MCTS is Black iff i is odd), each
game with a distinct nonzero seed `seed + i + 1` so AlphaBeta's root shuffle
varies lines. Counts MCTS wins, AlphaBeta wins, draws, and per-color splits.
Prints on stdout one `key value` per line:
```
games N
mcts_wins X
ab_wins Y
draws Z
mcts_black_wins A
mcts_white_wins B
ab_black_wins C
ab_white_wins D
```
Per-game progress on stderr (like `gomoku-bootstrap`). Register in
`CMakeLists.txt` (build, not install, link `gomoku_core`).

## Test discipline (TDD)
Write the new Python gates **first** (they fail: missing modules/tools), confirm
they fail, then implement `game_record.py`, `train.py`, and `headtohead.cpp` until
they pass. Keep every prior gate green throughout: `ctest` (75),
`test_hardware.py`, `test_gnn_roundtrip.py`, `test_forward.py`, `nix flake check`.

## Housekeeping
- `CMakeLists.txt`: add the `gomoku-headtohead` target.
- `docs/ARCHITECTURE.md`: add `game_record.py`, `train.py`, `test_game_record.py`,
  `test_train.py`, `test_headtohead.py`, and `gomoku-headtohead` to the training
  toolchain + tool tables; tick Increment 3 in `PLAN-stage3-training.md`.
- `AGENTS.md` Stage-3 section: note the Python GameRecord decoder and the
  head-to-head gate.
- `git add -N` every new file before `nix build`/`nix flake check`;
  `git diff --cached` stays empty (the sole documented exception —
  `git rm --cached training/uv.lock` — is already committed).

## Verification commands
```bash
nix develop -c cmake -S . -B build -G Ninja
nix develop -c cmake --build build
nix develop -c ctest --test-dir build                          # 75 stay green
cd training
scripts/sync.sh                                                # hardware-aware torch (already synced here)
uv run python tests/test_hardware.py                           # hermetic
uv run python tests/test_gnn_roundtrip.py --tool ../build/gomoku-dump-weights
uv run python tests/test_forward.py --tool ../build/gomoku-nn-eval
uv run python tests/test_game_record.py --tool ../build/gomoku-bootstrap
uv run python tests/test_train.py                              # needs torch
uv run python tests/test_headtohead.py --tool ../build/gomoku-headtohead \
    --model <some.gnn> --games 2                               # smoke; report-only
nix flake check
```

## Acceptance criteria
- `game_record.py` decodes `gomoku-bootstrap` output; round-trip + malformed
  rejection tests pass.
- `train.py` runs the SL→RL loop (a tiny end-to-end run produces `model.gnn` and
  at least one RL iteration) and uses `hardware.select_device()`.
- `gomoku-headtohead` plays the neural vs classic engine and reports the score;
  `test_headtohead.py` drives it.
- All prior gates green: `ctest` (75), `test_hardware.py`,
  `test_gnn_roundtrip.py`, `test_forward.py`, `nix flake check`; `-Werror` clean.
- New files `git add -N`; `git diff --cached` empty.
