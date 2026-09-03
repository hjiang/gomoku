"""Stage 3 Increment 3 training loop: SL (bootstrap) then RL (self-play).

Orchestration (strict order — self-play needs a trained model, so supervised
pre-training always runs first):

    gomoku-bootstrap --games N --depth D --time-ms T --seed S --jitter J --out sl.rec
    train SL on decode(sl.rec)                        # label-smoothed policy CE + value MSE
    export model.gnn
    for it in 1..rl_iters:
        gomoku-selfplay --model model.gnn --games M --sims K --time-ms T --seed S --out rl.rec
        train RL on decode(rl.rec)                    # visit-count policy CE + value MSE
        export model.gnn (plus a model.rl<N>.gnn copy)

The model is kept **in memory** and only exported; a ``.gnn`` is never reloaded
back into PyTorch (``gnn_format`` stays encode-only). Both SL and RL records
carry a *distribution* policy target, so both phases use the same cross-entropy
loss ``-(target * log_softmax).sum(1).mean()``.

Training samples are augmented on the fly with the 8-fold dihedral symmetry of
the board (planes and policy transformed by the same map, so the
(input, target) pair stays valid). ``--sl-jitter J`` randomizes the first J
bootstrap opening plies (center 5x5) so the supervised value targets are not
~100% Black wins. When ``--eval-tool`` is set and ``--eval-games > 0``, every
exported checkpoint is played against Classic-Hard and the best-by-winrate is
copied to ``model.best.gnn``.

Runtime device comes from ``hardware.select_device()`` (CUDA when available,
else CPU). Run ``training/scripts/sync.sh`` before the first run to install the
hardware-correct torch build.

Example:
    python train.py \\
        --bootstrap-tool ../build/gomoku-bootstrap \\
        --selfplay-tool ../build/gomoku-selfplay \\
        --outdir .pi/training --sl-games 40 --rl-iters 2
"""

from __future__ import annotations

import argparse
import concurrent.futures
import random
import struct
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

import export_gnn
import game_record
import hardware
import model as model_mod


# --- pure, import-safe helpers -------------------------------------------------
def flatten_records(games: list[list[game_record.PositionRecord]]) -> list[game_record.PositionRecord]:
    """Flatten a decoded stream (list of games) into a list of positions."""
    return [pos for game in games for pos in game]


def records_to_tensors(records: list[game_record.PositionRecord]):
    """Convert ``PositionRecord``s into ``(planes, policy, value)`` tensors.

    - ``planes``: ``(N, 4, 15, 15)`` float32, channel-major row-major — the
      exact layout ``BoardEncoder::encode`` writes, so a plain reshape works.
    - ``policy``: ``(N, 225)`` float32 distribution targets.
    - ``value``: ``(N, 1)`` float32 outcome targets.
    """
    n = len(records)
    if n == 0:
        raise ValueError("no records to convert")
    planes = torch.tensor([r.planes for r in records], dtype=torch.float32).view(
        n, 4, 15, 15
    )
    policy = torch.tensor([r.policy for r in records], dtype=torch.float32)
    value = torch.tensor([[r.value] for r in records], dtype=torch.float32)
    return planes, policy, value


def policy_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cross-entropy with a **distribution** target (not a class index).

    Identical for SL label-smoothed targets and RL visit-count targets:
    ``-(target * log_softmax(logits)).sum(1).mean()``.
    """
    log_probs = F.log_softmax(logits, dim=1)
    return -(target * log_probs).sum(dim=1).mean()


def value_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean-squared-error on the value head."""
    return F.mse_loss(pred, target)


def dihedral_transform(planes: torch.Tensor, policy: torch.Tensor,
                       k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one of the 8 dihedral (rotation + reflection) symmetries.

    The same spatial map is applied to ``planes`` (B,4,15,15) and ``policy``
    (B,225) so the (input, target) pair stays a valid training example:
    gomoku is invariant under the dihedral group of the square board. The
    value target needs no transform. ``k`` in 0..7 with ``rot = k % 4`` and
    ``flip = k >= 4`` (horizontal reflection, applied before rotation);
    ``k == 0`` is the identity.
    """
    rot = k % 4
    flip = k >= 4
    if flip:
        planes = torch.flip(planes, dims=(3,))
        policy = torch.flip(policy.reshape(-1, 15, 15), dims=(2,)).reshape(-1, 225)
    if rot:
        planes = torch.rot90(planes, rot, dims=(2, 3))
        policy = torch.rot90(policy.reshape(-1, 15, 15), rot, dims=(1, 2)).reshape(-1, 225)
    return planes, policy


def split_records(records: list[game_record.PositionRecord], val_frac: float,
                  seed: int) -> tuple[list[game_record.PositionRecord],
                                      list[game_record.PositionRecord]]:
    """Deterministically split ``records`` into ``(train, val)`` lists.

    ``val_frac`` must be in [0, 1); ``val_frac == 0`` returns all records as
    train and an empty val list. When ``val_frac > 0`` and more than one
    record is given, the val list is never empty, so small datasets still
    exercise the validation path. A local ``random.Random(seed)`` shuffles the
    indices (never the global RNG), so the split is reproducible under a fixed
    seed without disturbing torch's RNG state.
    """
    if not 0.0 <= val_frac < 1.0:
        raise ValueError(f"val_frac must be in [0, 1), got {val_frac}")
    if val_frac == 0.0 or not records:
        return list(records), []
    n = len(records)
    # Clamp up to 1 so round() cannot silently disable validation on small
    # datasets (e.g. n=8, val_frac=0.05 rounds to 0); n-1 keeps >= 1 train.
    n_val = min(max(round(n * val_frac), 1), n - 1)
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    val_idx = set(order[:n_val])
    train_records = [r for i, r in enumerate(records) if i not in val_idx]
    val_records = [r for i, r in enumerate(records) if i in val_idx]
    return train_records, val_records


def balanced_offsets(total: int, jobs: int) -> list[int]:
    """Start offsets of ``jobs`` balanced chunks whose sizes sum to ``total``.

    Chunk sizes differ by at most one, with the larger chunks first. ``jobs``
    is clamped to ``[1, total]``. Pre: ``total >= 1``.
    """
    if total < 1:
        raise ValueError("total must be >= 1")
    jobs = max(1, min(jobs, total))
    base, rem = divmod(total, jobs)
    offsets = []
    acc = 0
    for k in range(jobs):
        offsets.append(acc)
        acc += base + (1 if k < rem else 0)
    return offsets


def merge_rec_files(paths: list[Path], out: Path) -> None:
    """Concatenate several GameRecord streams into one, at the byte level.

    Each input must be a valid stream (same magic/version). The per-file 13-byte
    stream header is stripped and a single fresh header is written, so no full
    decode is needed — cheap for large self-play batches. All inputs are
    validated before ``out`` is touched, so a bad input cannot leave a partial
    ``out`` file.
    """
    bodies: list[bytes] = []
    for p in paths:
        with open(p, "rb") as src:
            data = src.read()
        if len(data) < game_record.HEADER_SIZE or \
                data[: len(game_record.MAGIC)] != game_record.MAGIC:
            raise ValueError(f"not a GameRecord stream: {p}")
        (version,) = struct.unpack_from("<I", data, len(game_record.MAGIC))
        if version != game_record.VERSION:
            raise ValueError(f"unsupported GameRecord version {version}: {p}")
        bodies.append(data[game_record.HEADER_SIZE:])

    header = game_record.MAGIC + struct.pack("<I", game_record.VERSION)
    with open(out, "wb") as dst:
        dst.write(header)
        for body in bodies:
            dst.write(body)


def train_epoch(model, optimizer, planes, policy, value, device, batch_size,
                value_weight: float = 1.0) -> float:
    """One epoch over on-device tensors in mini-batches. Returns mean loss.

    ``planes``/``policy``/``value`` must already be on ``device``. Each batch
    is augmented on the fly: every sample gets its own random 8-fold dihedral
    transform (drawn from the torch global RNG, seeded in main) applied to
    planes and policy together, so the (input, target) pair stays valid.
    Samples are grouped by transform (at most 8 batched calls per batch) so
    augmentation launches no per-sample kernels and no device syncs on CUDA.
    The loss is ``policy_cross_entropy + value_weight * value_mse``.
    """
    model.train()
    n = planes.shape[0]
    indices = torch.randperm(n, device=device)
    total = 0.0
    seen = 0
    for start in range(0, n, batch_size):
        idx = indices[start : start + batch_size]
        batch_planes = planes[idx]
        batch_policy = policy[idx]
        # One random dihedral symmetry per sample (torch global RNG). k is
        # drawn on the CPU and the batch grouped by transform (<= 8 batched
        # calls) so CUDA needs no per-sample kernels and no device sync.
        k = torch.randint(0, 8, (batch_planes.shape[0],))
        counts = torch.bincount(k, minlength=8)
        planes_out = torch.empty_like(batch_planes)
        policy_out = torch.empty_like(batch_policy)
        for ki in range(8):
            if int(counts[ki]) == 0:
                continue
            mask = (k == ki).to(batch_planes.device, non_blocking=True)
            aug_planes, aug_policy = dihedral_transform(
                batch_planes[mask], batch_policy[mask], ki)
            planes_out[mask] = aug_planes
            policy_out[mask] = aug_policy
        batch_planes = planes_out
        batch_policy = policy_out
        optimizer.zero_grad()
        logits, pred = model(batch_planes)
        loss = policy_cross_entropy(logits, batch_policy) + value_weight * value_mse(
            pred, value[idx]
        )
        loss.backward()
        optimizer.step()
        total += loss.item() * idx.numel()
        seen += idx.numel()
    return total / seen if seen else 0.0


# --- orchestration -------------------------------------------------------------
def run_tool(cmd: list[str]) -> None:
    """Run a C++ generator tool; raise with its stderr on failure."""
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{Path(cmd[0]).name} failed with exit {proc.returncode}\nstderr:\n{proc.stderr}"
        )


def generate_records(tool: str, *, games: int, jobs: int, base_seed: int,
                     extra: list[str], out: Path, label: str) -> None:
    """Generate ``games`` games with ``tool``, split across ``jobs`` processes.

    Each process writes a ``<out>.part<K>`` stream with a chunk of games and a
    distinct base seed; the parts are then merged into ``out``. ``jobs == 1``
    runs a single process straight into ``out`` (no temporary part files).
    """
    jobs = max(1, min(jobs, games))
    if jobs == 1:
        run_tool([tool, *extra, "--games", str(games), "--seed", str(base_seed),
                  "--out", str(out)])
        return

    offsets = balanced_offsets(games, jobs)
    parts: list[Path] = []
    commands: list[list[str]] = []
    for k in range(jobs):
        chunk = (offsets[k + 1] if k + 1 < jobs else games) - offsets[k]
        part = out.with_name(f"{out.name}.part{k}")
        parts.append(part)
        commands.append([tool, *extra, "--games", str(chunk),
                         "--seed", str(base_seed + offsets[k]), "--out", str(part)])

    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        list(pool.map(run_tool, commands))
    merge_rec_files(parts, out)
    for p in parts:
        p.unlink(missing_ok=True)
    print(f"{label}: merged {games} games from {jobs} parallel runs into {out}",
          file=sys.stderr)


# Fields gomoku-headtohead prints as "key value" lines on stdout (mirrors
# tests/test_headtohead.parse_output).
_H2H_FIELDS = ("games", "mcts_wins", "ab_wins", "draws", "mcts_black_wins",
               "mcts_white_wins", "ab_black_wins", "ab_white_wins")


def run_headtohead(eval_tool: str, path: Path, games: int, label: str):
    """Run ``gomoku-headtohead`` on ``path`` (Hard presets) and parse its output.

    Returns the parsed field dict, or ``None`` if the tool failed or its output
    was malformed. Prints one ``EVAL <label>: ...`` summary line to stderr.
    """
    try:
        proc = subprocess.run([eval_tool, "--model", str(path), "--games", str(games)],
                              capture_output=True, text=True)
    except OSError as exc:
        print(f"EVAL {label}: gomoku-headtohead failed to run: {exc}",
              file=sys.stderr)
        return None
    if proc.returncode != 0:
        print(f"EVAL {label}: gomoku-headtohead failed with exit {proc.returncode}: "
              f"{proc.stderr.strip()}", file=sys.stderr)
        return None
    fields = {}
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] in _H2H_FIELDS:
            fields[parts[0]] = int(parts[1])
    if any(f not in fields for f in _H2H_FIELDS):
        print(f"EVAL {label}: malformed gomoku-headtohead output:\n{proc.stdout}",
              file=sys.stderr)
        return None
    decided = fields["mcts_wins"] + fields["ab_wins"]
    winrate = fields["mcts_wins"] / decided if decided else 0.0
    print(f"EVAL {label}: {fields['mcts_wins']}/{decided} "
          f"(black {fields['mcts_black_wins']}, white {fields['mcts_white_wins']}) "
          f"winrate {winrate:.1%}", file=sys.stderr)
    return fields


def train_on_records(model, records, *, device, epochs, batch_size, optimizer,
                     value_weight, label: str, val_frac: float = 0.0,
                     val_seed: int = 0) -> list[dict[str, float]]:
    """Build tensors from decoded records and train for ``epochs`` epochs.

    ``records`` are split deterministically (``split_records``) into train/val;
    the ``optimizer`` is caller-owned so its momentum state survives across
    phases and RL iterations. When a validation set exists, each epoch log line
    also reports the UNtransformed val policy-CE and value-MSE computed in
    eval mode. Returns one dict per epoch: ``loss`` plus ``val_ce``/``val_mse``
    when a val split exists.
    """
    train_records, val_records = split_records(records, val_frac, val_seed)
    planes, policy, value = records_to_tensors(train_records)
    planes = planes.to(device)
    policy = policy.to(device)
    value = value.to(device)

    has_val = bool(val_records)
    if has_val:
        val_planes, val_policy, val_value = records_to_tensors(val_records)
        val_planes = val_planes.to(device)
        val_policy = val_policy.to(device)
        val_value = val_value.to(device)

    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, optimizer, planes, policy, value, device,
                           batch_size, value_weight)
        if has_val:
            model.eval()
            with torch.no_grad():
                val_logits, val_pred = model(val_planes)
                val_ce = policy_cross_entropy(val_logits, val_policy).item()
                val_mse = value_mse(val_pred, val_value).item()
            model.train()
            history.append({"loss": loss, "val_ce": val_ce, "val_mse": val_mse})
            print(f"{label} epoch {epoch}: loss {loss:.4f} "
                  f"val_ce {val_ce:.4f} val_mse {val_mse:.4f}", file=sys.stderr)
        else:
            history.append({"loss": loss})
            print(f"{label} epoch {epoch}: loss {loss:.4f}", file=sys.stderr)
    return history


def export_model(model, path: Path) -> None:
    """Write the current in-memory model as a ``.gnn`` file."""
    model.eval()  # BatchNorm runs on its running stats in the exported file
    path.write_bytes(export_gnn.export_gnn(model))
    print(f"exported {path}", file=sys.stderr)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-tool", required=True,
                        help="path to gomoku-bootstrap (SL generator)")
    parser.add_argument("--selfplay-tool", required=True,
                        help="path to gomoku-selfplay (RL generator)")
    parser.add_argument("--outdir", default=".pi/training",
                        help="directory for .rec/.gnn artifacts")
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--sl-games", type=int, default=40)
    parser.add_argument("--sl-depth", type=int, default=2)
    parser.add_argument("--sl-time-ms", type=int, default=200)
    parser.add_argument("--sl-epochs", type=int, default=2)
    parser.add_argument("--sl-batch-size", type=int, default=64)
    parser.add_argument("--sl-lr", type=float, default=1e-3)

    parser.add_argument("--rl-iters", type=int, default=2)
    parser.add_argument("--rl-games", type=int, default=10)
    parser.add_argument("--rl-sims", type=int, default=100)
    parser.add_argument("--rl-time-ms", type=int, default=500)
    parser.add_argument("--rl-epochs", type=int, default=1)
    parser.add_argument("--rl-batch-size", type=int, default=64)
    parser.add_argument("--rl-lr", type=float, default=1e-4)

    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--export-every", type=int, default=1,
                        help="write a model.rl<N>.gnn copy every N RL iterations")
    parser.add_argument("--jobs", type=int, default=1,
                        help="parallel generation processes per SL/RL round "
                             "(single-threaded C++ tools; games are independent)")
    parser.add_argument("--val-frac", type=float, default=0.05,
                        help="fraction of records held out per phase as a "
                             "deterministic validation split (0 disables; "
                             "val_ce/val_mse then appear on each epoch line)")
    parser.add_argument("--sl-jitter", type=int, default=2,
                        help="randomize the first N bootstrap opening plies "
                             "inside the center 5x5 (forwarded to "
                             "gomoku-bootstrap as --jitter)")
    parser.add_argument("--eval-tool", default=None,
                        help="path to gomoku-headtohead; when set with "
                             "--eval-games > 0, each exported checkpoint is "
                             "evaluated at Hard presets and the "
                             "best-by-winrate is written to model.best.gnn")
    parser.add_argument("--eval-games", type=int, default=0,
                        help="games per head-to-head eval (0 disables; the tool "
                             "defaults already are Hard: sims 1600 / depth 6 / "
                             "2000 ms)")
    args = parser.parse_args(argv)

    if not 1 <= args.num_blocks <= 1024 or not 1 <= args.channels <= 1024:
        parser.error("--num-blocks/--channels must be in [1, 1024]")
    if args.export_every < 1:
        parser.error("--export-every must be >= 1")
    if args.jobs < 1:
        parser.error("--jobs must be >= 1")
    if args.sl_jitter < 0:
        parser.error("--sl-jitter must be >= 0")
    if not 0.0 <= args.val_frac < 1.0:
        parser.error("--val-frac must be in [0, 1)")
    if args.eval_games < 0:
        parser.error("--eval-games must be >= 0")
    if args.eval_games > 0 and not args.eval_tool:
        parser.error("--eval-games > 0 requires --eval-tool")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = hardware.select_device()
    torch.manual_seed(args.seed)
    print(f"device: {device}", file=sys.stderr)

    model = model_mod.GomokuNet(num_blocks=args.num_blocks, channels=args.channels)
    model = model.to(device)

    # Two persistent optimizers: one per phase. Reusing a single Adam across
    # the RL iterations keeps the momentum state alive instead of discarding
    # it (and the LR) every round.
    sl_optimizer = torch.optim.Adam(model.parameters(), lr=args.sl_lr)
    rl_optimizer = torch.optim.Adam(model.parameters(), lr=args.rl_lr)

    best_winrate = -1.0
    best_path: Path | None = None

    def evaluate_checkpoint(path: Path, label: str) -> None:
        """Run the head-to-head eval for ``path``; update ``model.best.gnn``."""
        nonlocal best_winrate, best_path
        if args.eval_games <= 0:
            return
        fields = run_headtohead(args.eval_tool, path, args.eval_games, label)
        if fields is None:
            return
        decided = fields["mcts_wins"] + fields["ab_wins"]
        winrate = fields["mcts_wins"] / decided if decided else 0.0
        # Strictly greater: ties keep the earlier best.
        if winrate > best_winrate:
            best_winrate = winrate
            best_path = path
            (outdir / "model.best.gnn").write_bytes(path.read_bytes())
            print(f"EVAL {label}: new best ({winrate:.1%}) -> model.best.gnn",
                  file=sys.stderr)

    # --- SL: supervised pre-training from bootstrap records -------------------
    sl_rec = outdir / "sl.rec"
    generate_records(
        args.bootstrap_tool, games=args.sl_games, jobs=args.jobs,
        base_seed=args.seed,
        extra=["--depth", str(args.sl_depth), "--time-ms", str(args.sl_time_ms),
               "--jitter", str(args.sl_jitter)],
        out=sl_rec, label="SL",
    )
    sl_records = flatten_records(game_record.decode_file(str(sl_rec)))
    print(f"SL: {len(sl_records)} positions from {sl_rec}", file=sys.stderr)
    train_on_records(
        model, sl_records, device=device, epochs=args.sl_epochs,
        batch_size=args.sl_batch_size, optimizer=sl_optimizer,
        value_weight=args.value_weight, label="SL",
        val_frac=args.val_frac, val_seed=args.seed,
    )
    export_model(model, outdir / "model.sl.gnn")
    export_model(model, outdir / "model.gnn")
    evaluate_checkpoint(outdir / "model.sl.gnn", "sl")

    # --- RL: self-play with the current model, then train ---------------------
    for it in range(1, args.rl_iters + 1):
        rl_rec = outdir / f"rl.it{it}.rec"
        generate_records(
            args.selfplay_tool, games=args.rl_games, jobs=args.jobs,
            base_seed=args.seed + it,
            extra=["--model", str(outdir / "model.gnn"),
                   "--sims", str(args.rl_sims), "--time-ms", str(args.rl_time_ms)],
            out=rl_rec, label=f"RL{it}",
        )
        rl_records = flatten_records(game_record.decode_file(str(rl_rec)))
        print(f"RL iteration {it}: {len(rl_records)} positions from {rl_rec}",
              file=sys.stderr)
        # Anchor RL on the supervised bootstrap: training only on the fresh
        # self-play batch lets the noisy low-sim visit-count targets overwrite
        # the alpha-beta teacher policy (catastrophic forgetting). Mixing the
        # SL records back in every iteration keeps the policy grounded.
        # NOTE: this re-splits the combined list, so the SL-phase holdout
        # re-enters the RL training set and the RL val split differs from the
        # SL one. That is fine today - val is advisory only (no early stop or
        # model selection reads it). If val is ever promoted to a gate, carry
        # the SL holdout forward and exclude it here.
        train_on_records(
            model, sl_records + rl_records, device=device, epochs=args.rl_epochs,
            batch_size=args.rl_batch_size, optimizer=rl_optimizer,
            value_weight=args.value_weight, label=f"RL{it}",
            val_frac=args.val_frac, val_seed=args.seed + it,
        )
        # The latest model always goes to model.gnn (the next self-play round
        # loads it); dated copies land per --export-every.
        export_model(model, outdir / "model.gnn")
        if it % args.export_every == 0 or it == args.rl_iters:
            checkpoint = outdir / f"model.rl{it}.gnn"
            export_model(model, checkpoint)
            evaluate_checkpoint(checkpoint, f"rl{it}")

    print(f"done: latest model at {outdir / 'model.gnn'}")
    if args.eval_games > 0 and best_path is not None:
        print(f"done: best-by-eval at {outdir / 'model.best.gnn'} "
              f"(winrate {best_winrate:.1%}, from {best_path.name})",
              file=sys.stderr)


if __name__ == "__main__":
    main()
