# Gomoku — Player vs AI

An offline, single-player **Gomoku** (five-in-a-row) game for Linux: you play
Black against a computer opponent on a 15×15 board. No network, no accounts,
no servers — everything runs locally.

- **Two switchable AI engines**
  - **Classic** — a deterministic alpha-beta search over a pattern heuristic.
  - **Neural** — Monte-Carlo tree search guided by a trained policy/value
    network (AlphaZero-style). A trained model is **bundled**, so it works out
    of the box.
- **Difficulty levels** (Easy / Medium / Hard) tune the search budget for both
  engines.
- **Pure C++ runtime** (C++23 + Qt 6). Python is used *only* for the offline
  training toolchain, never at runtime.

---

## Building & running

Everything is reproducible through Nix — `g++` is not on `PATH` outside the
dev shell, so always prefix commands with `nix develop -c`:

```bash
nix develop -c cmake -S . -B build -G Ninja   # configure
nix develop -c cmake --build build            # build (-Wall -Wextra -Werror)
nix develop -c ctest --test-dir build         # run the 75 tests
nix run .#                                    # launch the game
```

`nix run .#` runs the game from the Nix store, with the trained model installed
at `share/gomoku/model.gnn`. During development you can run `build/gomoku`
directly, but the bundled model is not on that binary's lookup path — either
set `GOMOKU_MODEL_PATH=resources/model.gnn`, or copy the model next to the
binary (`cp resources/model.gnn build/`).

## Playing

- Click an empty intersection to place your stone (you are **Black** and move
  first; the AI is **White**).
- Use the toolbar to start a **New Game**, **Undo**, change **Difficulty**, or
  switch the **Engine** between *Classic* and *Neural*.
- The **Neural** engine needs a model. The game ships one
  (`resources/model.gnn`); if no model can be loaded, the *Neural* option is
  greyed out rather than silently falling back to the heuristic.
  To use a different model, set the environment variable before launching:

  ```bash
  GOMOKU_MODEL_PATH=/path/to/model.gnn nix run .#
  ```

## Testing

```bash
nix develop -c ctest --test-dir build   # 75 unit tests (74 Catch2 + 1 Qt Test)
nix flake check                         # builds the package + runs the tests
```

---

## Training a model

You only need this if you want to train a **new** model (e.g. a stronger one,
or a different architecture). The shipped model in `resources/model.gnn` was
produced with exactly this pipeline.

### How it works

Training is a two-phase AlphaZero-style loop, orchestrated by
`training/train.py`:

1. **Supervised bootstrap (SL)** — a headless tool (`gomoku-bootstrap`) plays
   the Classic engine against itself and records games. The network is trained
   to imitate the Classic engine's moves (softened) and predict the outcome.
   This is a cheap warm start.
2. **Self-play RL** — a second tool (`gomoku-selfplay`) plays the *current*
   network against itself (MCTS + root noise) and records games. The network
   is then trained on those games (visit-count policy targets + outcome value
   targets), with the bootstrap data mixed back in so it does not forget the
   teacher.

Each phase runs the C++ tools in parallel (`--jobs`), merges their record
streams, and trains the PyTorch network; the model is exported as a `.gnn`
file at every stage.

### Prerequisites

1. **Build the C++ tools** (they are part of the normal build):

   ```bash
   nix develop -c cmake --build build
   ```

   This produces `build/gomoku-bootstrap`, `build/gomoku-selfplay`,
   `build/gomoku-headtohead`, and friends.

2. **Install PyTorch** the hardware-aware way (never a bare `uv sync` — that
   pulls a ~4.6 GB CUDA build even on CPU-only machines):

   ```bash
   cd training
   ./scripts/sync.sh
   ```

   `scripts/sync.sh` auto-detects CPU vs NVIDIA GPU and installs the matching
   wheel. On a CPU-only machine this is the small CPU build.

### Quick smoke run

Verify the whole pipeline end-to-end with tiny budgets before a long run:

```bash
cd training
uv run python train.py \
    --bootstrap-tool ../build/gomoku-bootstrap \
    --selfplay-tool ../build/gomoku-selfplay \
    --outdir .pi/smoke \
    --sl-games 8 --sl-depth 2 --sl-epochs 1 \
    --rl-iters 1 --rl-games 8 --rl-sims 50 \
    --jobs 4
```

It should print `SL: …`, `RL1: …` and finish with
`done: best model at .pi/smoke/model.gnn`.

### A real training run

The values below are the ones used to train the shipped model (≈ 1–2 hours on
a 16-core CPU-only machine). Raise `--sl-games`/`--sl-epochs` for a better
imitation, and `--rl-iters`/`--rl-sims` for stronger self-play:

```bash
cd training
uv run python train.py \
    --bootstrap-tool ../build/gomoku-bootstrap \
    --selfplay-tool ../build/gomoku-selfplay \
    --outdir .pi/training \
    --num-blocks 4 --channels 16 \
    --seed 20260813 \
    --sl-games 512 --sl-depth 6 --sl-time-ms 2000 \
    --sl-epochs 6 --sl-batch-size 64 --sl-lr 1e-3 \
    --rl-iters 5 --rl-games 160 --rl-sims 800 --rl-time-ms 5000 \
    --rl-epochs 2 --rl-batch-size 64 --rl-lr 1e-4 \
    --value-weight 1.0 --export-every 1 \
    --jobs 16
```

Notes:

- `--jobs` parallelizes the single-threaded C++ generators; set it to your core
  count (or a bit less, to leave headroom for the Python trainer).
- `--sl-depth 6 --sl-time-ms 2000` makes the bootstrap teacher Classic-Hard
  strength (the search is time-bound, so the depth cap is free extra headroom).
- `--rl-sims` should roughly match the evaluation budget: the Neural engine
  gets 1600 sims but a 2000 ms deadline, which on CPU is ≈ 800 effective sims.
- The run writes `model.sl.gnn`, `model.rl<N>.gnn`, and always the latest
  `model.gnn` into `--outdir`. The `.gnn` model is never reloaded into PyTorch;
  training keeps it in memory and only exports.

### Evaluate a model (head-to-head vs Classic)

Measure the trained model's strength by playing it against the Classic engine
at **Hard** (depth 6 / 2000 ms):

```bash
cd training
uv run python tests/test_headtohead.py \
    --tool ../build/gomoku-headtohead \
    --model ../.pi/training/model.rl3.gnn \
    --games 20
```

Add `--min-winrate 0.5` to turn it into a hard gate (it fails if the Neural
engine wins fewer than half of the decided games). The shipped model scores
**50%** — it wins every game as Black (first mover) and, like Classic, cannot
overcome freestyle gomoku's decisive first-mover advantage as White.

### Ship a model

Once you have a model you like, copy it into the game's data path so the
Neural engine uses it by default:

```bash
cp .pi/training/model.rl3.gnn resources/model.gnn   # from the repo root
nix run .#                                           # now uses the new model
```

`resources/model.gnn` is the single file the game loads (see
`src/ui/MainWindow.cpp`); `GOMOKU_MODEL_PATH` always overrides it.

## Project layout

```
src/core/    Qt-free game rules + both AI engines (unit-tested)
src/ui/      Qt 6 widgets (thin presentation shell)
src/tools/   headless training/eval tools (developer tooling, never installed)
training/    Python training toolchain (uv-managed, PyTorch)
resources/   the shipped trained model
docs/        requirements, architecture, and stage plans
```

See [`docs/REQUIREMENTS.md`](docs/REQUIREMENTS.md) for the full requirements
and [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the design.
