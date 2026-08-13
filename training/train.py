"""Stage 3 Increment 3 training loop: SL (bootstrap) then RL (self-play).

Orchestration (strict order — self-play needs a trained model, so supervised
pre-training always runs first):

    gomoku-bootstrap --games N --depth D --time-ms T --seed S --out sl.rec
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


def train_epoch(model, optimizer, planes, policy, value, device, batch_size,
                value_weight: float = 1.0) -> float:
    """One epoch over on-device tensors in mini-batches. Returns mean loss.

    ``planes``/``policy``/``value`` must already be on ``device``. The loss is
    ``policy_cross_entropy + value_weight * value_mse``.
    """
    model.train()
    n = planes.shape[0]
    indices = torch.randperm(n, device=device)
    total = 0.0
    seen = 0
    for start in range(0, n, batch_size):
        idx = indices[start : start + batch_size]
        optimizer.zero_grad()
        logits, pred = model(planes[idx])
        loss = policy_cross_entropy(logits, policy[idx]) + value_weight * value_mse(
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


def train_on_records(model, records, *, device, epochs, batch_size, lr,
                     value_weight, label: str) -> None:
    """Build tensors from decoded records and train for ``epochs`` epochs."""
    planes, policy, value = records_to_tensors(records)
    planes = planes.to(device)
    policy = policy.to(device)
    value = value.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, optimizer, planes, policy, value, device,
                           batch_size, value_weight)
        print(f"{label} epoch {epoch}: loss {loss:.4f}", file=sys.stderr)


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
    args = parser.parse_args(argv)

    if not 1 <= args.num_blocks <= 1024 or not 1 <= args.channels <= 1024:
        parser.error("--num-blocks/--channels must be in [1, 1024]")
    if args.export_every < 1:
        parser.error("--export-every must be >= 1")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = hardware.select_device()
    torch.manual_seed(args.seed)
    print(f"device: {device}", file=sys.stderr)

    model = model_mod.GomokuNet(num_blocks=args.num_blocks, channels=args.channels)
    model = model.to(device)

    # --- SL: supervised pre-training from bootstrap records -------------------
    sl_rec = outdir / "sl.rec"
    run_tool([
        args.bootstrap_tool, "--games", str(args.sl_games),
        "--depth", str(args.sl_depth), "--time-ms", str(args.sl_time_ms),
        "--seed", str(args.seed), "--out", str(sl_rec),
    ])
    sl_records = flatten_records(game_record.decode_file(str(sl_rec)))
    print(f"SL: {len(sl_records)} positions from {sl_rec}", file=sys.stderr)
    train_on_records(
        model, sl_records, device=device, epochs=args.sl_epochs,
        batch_size=args.sl_batch_size, lr=args.sl_lr, value_weight=args.value_weight,
        label="SL",
    )
    export_model(model, outdir / "model.sl.gnn")
    export_model(model, outdir / "model.gnn")

    # --- RL: self-play with the current model, then train ---------------------
    for it in range(1, args.rl_iters + 1):
        rl_rec = outdir / f"rl.it{it}.rec"
        run_tool([
            args.selfplay_tool, "--model", str(outdir / "model.gnn"),
            "--games", str(args.rl_games), "--sims", str(args.rl_sims),
            "--time-ms", str(args.rl_time_ms), "--seed", str(args.seed + it),
            "--out", str(rl_rec),
        ])
        rl_records = flatten_records(game_record.decode_file(str(rl_rec)))
        print(f"RL iteration {it}: {len(rl_records)} positions from {rl_rec}",
              file=sys.stderr)
        train_on_records(
            model, rl_records, device=device, epochs=args.rl_epochs,
            batch_size=args.rl_batch_size, lr=args.rl_lr,
            value_weight=args.value_weight, label=f"RL{it}",
        )
        # The latest model always goes to model.gnn (the next self-play round
        # loads it); dated copies land per --export-every.
        export_model(model, outdir / "model.gnn")
        if it % args.export_every == 0 or it == args.rl_iters:
            export_model(model, outdir / f"model.rl{it}.gnn")

    print(f"done: best model at {outdir / 'model.gnn'}")


if __name__ == "__main__":
    main()
