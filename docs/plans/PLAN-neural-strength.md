# PLAN: stronger neural model (training-process fixes)

## Problem

The shipped model (`resources/model.gnn`, run `.pi/inc4-training3/model.rl3.gnn`) is
weak - a human beats it. Code review found **no correctness bug** (data verified
byte-level, C++ forward gated vs torch, MCTS math correct). The causes, with
evidence from `/home/hjiang/code/gomoku`:

1. **Tiny dataset**: 512 SL games = 9,454 positions for a ~180k-param net; SL
   loss 1.35 and still falling when training stopped (label-smoothed entropy
   floor ~0.86).
2. **Degenerate value targets**: SL games ended 510/512 Black wins, so the
   value head learned move parity, not position strength. Probes of the shipped
   model: empty board v=+0.83 (Black to move), one stone played v=-0.89 (White
   to move). In every H2H log the neural engine won **0 games as White** -
   and the human user always plays Black against it.
3. **Policy head is tactically blind**: probes on legal positions give the
   win-completing cell 0.4-0.7% and the forced block 0.1-0.8% policy mass.
4. **Model selection by noise**: 20-game H2H evals (+-20pt CI); the shipped
   checkpoint was hand-picked (rl3=50%, while the final rl5=25%).

Scope of this plan: change what the *training pipeline* feeds and selects so a
retrain on a bigger machine produces a genuinely stronger model. No changes to
the `.gnn` format, the GameRecord binary format, the MCTS algorithm, or the UI.

## Changes (priority order)

### 1. C++ bootstrap opening jitter (fixes cause 2)

Balanced value targets need games where White wins. Cheapest correct lever:
randomize the first N opening plies of each bootstrap game within the central
region; the teachers then play from slightly-off openings and White wins a real
fraction.

Files:
- `src/core/SearchEngine.hpp`: add to `SearchParams`:
  `int openingJitter = 0;` (first N plies random) and
  `int openingRadius = 2;` (region half-width; 2 = center 5x5).
- `src/core/SelfPlay.cpp` `generateBootstrapGame`: seed a
  `std::mt19937_64 rng(seed)`; for `ply < openingJitter` pick a uniform random
  empty cell with `|r-7|<=radius && |c-7|<=radius` instead of calling
  AlphaBeta. **Keep recording every position** (including jittered ones) with
  the usual label-smoothed target on the played move - this preserves the
  documented parity contract ("position i: Black to move iff i even"), the
  binary format, and all existing tests. Add `#include <random>`.
  `generateSelfPlayGame` is unchanged (Dirichlet noise already diversifies).
- `src/tools/bootstrap.cpp`: new `--jitter N` flag (default 0), wired to
  `params.openingJitter`.

Tests (`tests/test_selfplay.cpp`):
- default params (`openingJitter=0`) still produce byte-identical records per
  seed (already covered by the existing determinism test - keep it as the
  back-compat pin);
- with `openingJitter=2`: deterministic per seed; both opening moves inside the
  center 5x5; `checkGameConsistency` passes (it already validates the played
  move has policy mass and values match the outcome).

Sanity gate after a real generation batch: decode `sl.rec` in Python and print
the winner histogram (value of position 0). Expect Black share to drop from
99.6% to <=~80%. One-off script; not committed.

### 2. 8-fold dihedral augmentation in the trainer (multiplies cause-1 fix by 8)

Gomoku is invariant under 4 rotations x 2 reflections. Apply a random dihedral
transform per sample per epoch - same map on planes and policy so the
(input, target) pair stays valid. Value target unchanged.

Files:
- `training/train.py`: new pure helper
  `dihedral_transform(planes (B,4,15,15), policy (B,225), k in 0..7)`
  using `torch.rot90`/`torch.flip` on spatial dims (policy viewed as
  (B,15,15) then flattened). `train_epoch` draws `k ~ randint(0,8)` per sample
  each epoch (torch global RNG; seeded in main already). Validation loss is
  computed on **untransformed** data.
- Both SL and RL phases get it automatically (both go through
  `train_on_records`/`train_epoch`).

Tests (`training/tests/test_train.py`): hand-computed anchors for k=1 (pure
rot90: cell (r,c) -> (14-c, r)) and k=4 (pure flip: (r,c) -> (r, 14-c));
plane-marker cell always equals policy-argmax cell after any k; k=0 identity;
transform is a permutation (sums preserved); augmented `train_epoch` still
steps parameters.

### 3. Holdout validation + best-checkpoint selection (fixes cause 4)

Files:
- `training/train.py`:
  - `--val-frac` (default 0.05): deterministic split of SL records and of each
    RL batch; per-epoch log line gains val policy-CE and val value-MSE
    (`model.eval()` + `no_grad`, then back to train).
  - `--eval-tool <path to gomoku-headtohead>` + `--eval-games N` (default 0 =
    off). After each export (`model.sl.gnn`, `model.rl<N>.gnn`) run the tool
    at Hard presets (sims 1600 / depth 6 / 2000 ms), parse its `key value`
    stdout (local parser mirroring `tests/test_headtohead.py`), log
    `EVAL rl<N>: X/Y (black b, white w)`, and copy the best checkpoint by win
    rate to `model.best.gnn`. `model.gnn` keeps its current meaning (latest,
    feeds the next self-play round).
  - Give each phase its own `torch.optim.Adam` (SL and RL use different
    learning rates); create the RL one once in `main` and pass it into
    `train_on_records` so RL momentum survives across rounds (previously
    discarded each round).
- `README.md`: ship-from `model.best.gnn`; document new flags.

Note for runbook: each eval game takes ~1-3 min at Hard budgets, so
`--eval-games 24` adds ~30-70 min per checkpoint. Worth it; it is the only
strength signal with a floor under its variance.

### 4. Optional, do NOT block the training run on these

- **Lower SL label smoothing** `eps 0.1 -> 0.05`: thread a
  `float bootstrapEps` through SearchParams -> `labelSmoothedPolicy`. 10% mass
  spread over ~200 cells is heavy; with augmentation it matters less.
- **NN eval speedup** (`src/core/NeuralNet.cpp`/`TensorOps.hpp`): preallocated
  `thread_local` scratch buffers instead of per-call `std::vector`
  reallocations. Measured 2.2 ms/sim today (=> ~900 effective sims in the Hard
  2000 ms budget vs the nominal 1600); a 1.5-2x win raises playing strength at
  fixed budget and cuts self-play wall time. Must not change any math - gated
  by `tests/test_neural.cpp`, ctest, and `tests/test_forward.py`.
- **MCTS child neighborhood restriction** (expand only empty cells within
  radius 2 of any stone): concentrates visits on plausible moves at fixed sim
  budget; standard for 15x15 gomoku. Changes self-play policy targets (zeros
  outside the neighborhood are fine - targets still sum to 1). Independent of
  retraining; evaluate separately via headtohead.

Net size stays N=4/C=16: runtime is time-bound, and a bigger net (e.g. C=32,
~4x eval cost) would *reduce* effective sims per move.

## Verification

In-repo gates (must all pass before handing the run off):
- `nix develop -c cmake --build build && nix develop -c ctest --test-dir build`
- `uv run python tests/test_train.py`
- `uv run python tests/test_game_record.py --tool ../build/gomoku-bootstrap`
  (stream format unchanged by jitter)
- `uv run python tests/test_forward.py --tool ../build/gomoku-nn-eval`
  (only if item 4 speedup is attempted)
- Smoke orchestration: `uv run python train.py --bootstrap-tool ../build/gomoku-bootstrap
  --selfplay-tool ../build/gomoku-selfplay --outdir ../.pi/smoke --sl-games 4
  --rl-iters 1 --jobs 2` completes; decoding its `sl.rec` shows non-teacher
  first moves.

Success criteria for the full run (measured on the training machine):
- SL val loss clearly plateaued before the last epoch.
- Bootstrap winner histogram: Black <= ~80% of games (from 99.6%).
- `uv run python tests/test_headtohead.py --tool ../build/gomoku-headtohead
  --model <model.best.gnn> --games 100 --min-winrate 0.6` passes, **and**
  `mcts_white_wins >= 5` of the ~50 White games (today: 0). White strength is
  the user-facing metric - the human always plays Black.
- Re-run the policy probes (win-in-1 completion, forced block): the chosen
  cells should now carry visible policy mass (>5%) even before search.

## Runbook for the training machine

Setup (per AGENTS.md): `cd training && ./scripts/sync.sh` (hardware-aware
torch; CUDA is picked up automatically via `hardware.select_device()`), and
build the C++ tools (`nix develop -c cmake --build build`).

Preset A - 16-core CPU-only, roughly overnight:
```bash
cd training
uv run python train.py \
  --bootstrap-tool ../build/gomoku-bootstrap \
  --selfplay-tool ../build/gomoku-selfplay \
  --eval-tool ../build/gomoku-headtohead \
  --outdir ../.pi/train-v2 --num-blocks 4 --channels 16 --seed 20260814 \
  --sl-games 4096 --sl-depth 6 --sl-time-ms 1000 --sl-jitter 2 \
  --sl-epochs 16 --sl-batch-size 64 --sl-lr 1e-3 \
  --rl-iters 4 --rl-games 400 --rl-sims 800 --rl-time-ms 5000 \
  --rl-epochs 2 --rl-batch-size 64 --rl-lr 1e-4 \
  --eval-games 24 --jobs 14
```

Preset B - CUDA machine: same but `--sl-games 12288 --sl-epochs 30
--rl-games 800`, `--jobs` = physical cores (generation is CPU-bound even when
training uses the GPU).

Rough wall times: bootstrap ~18 moves/game at <=1 s/move; self-play ~22
moves/game x 800 sims x ~2.2 ms; expect a few hours for SL generation and
~20-40 min per RL iteration of generation+training on 14 jobs, plus eval time.

Ship: `cp <outdir>/model.best.gnn resources/model.gnn` from the repo root
(tracked file - replace in place; no `git add -N` needed), then re-run the
100-game gate above.

## Explicit non-goals

- No `.gnn` format, GameRecord format, MCTS, UI, or difficulty-preset changes.
- No attempt to make the neural engine beat Classic as White at equal budget -
  freestyle gomoku's first-mover advantage is real; the target is "no longer
  trivially beaten by a human", measured as above.
