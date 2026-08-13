#!/usr/bin/env python3
"""Head-to-head gate: the trained neural engine vs the classic engine.

Drives ``gomoku-headtohead`` (the C++ tool that plays MCTS-with-model vs
Classic-Hard alpha-beta) and reports the neural engine's score. Report-only by
default; pass ``--min-winrate`` to turn it into a hard gate (fails when the
MCTS win rate over decided games falls short). Increment 4 sets the threshold.

Usage:  python tests/test_headtohead.py --tool ../build/gomoku-headtohead \
            --model <model.gnn> --games N [--min-winrate 0.5]
"""

import argparse
import subprocess
import sys

FIELDS = (
    "games",
    "mcts_wins",
    "ab_wins",
    "draws",
    "mcts_black_wins",
    "mcts_white_wins",
    "ab_black_wins",
    "ab_white_wins",
)


def parse_output(stdout: str) -> dict:
    """Parse the tool's ``key value`` stdout into an int dict."""
    result = {}
    for line in stdout.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] in FIELDS:
            result[parts[0]] = int(parts[1])
    missing = [f for f in FIELDS if f not in result]
    if missing:
        raise AssertionError(
            f"headtohead output missing fields {missing}:\n{stdout}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool", required=True, help="path to gomoku-headtohead")
    parser.add_argument("--model", required=True, help="path to a .gnn model")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--sims", type=int, default=1600,
                        help="MCTS simulation budget (Classic Hard uses 1600)")
    parser.add_argument("--depth", type=int, default=6,
                        help="alpha-beta depth (Classic Hard uses 6)")
    parser.add_argument("--time-ms", type=int, default=2000,
                        help="soft deadline for both engines (Classic Hard)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--min-winrate", type=float, default=None,
                        help="fail unless MCTS win rate over decided games "
                             ">= this fraction")
    args = parser.parse_args()

    proc = subprocess.run(
        [args.tool, "--model", args.model, "--games", str(args.games),
         "--sims", str(args.sims), "--depth", str(args.depth),
         "--time-ms", str(args.time_ms), "--seed", str(args.seed)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        sys.exit(f"gomoku-headtohead failed with exit {proc.returncode}")

    r = parse_output(proc.stdout)
    decided = r["mcts_wins"] + r["ab_wins"]
    winrate = r["mcts_wins"] / decided if decided else float("nan")

    print(f"games={r['games']} mcts_wins={r['mcts_wins']} ab_wins={r['ab_wins']} "
          f"draws={r['draws']}")
    print(f"color split: mcts_black={r['mcts_black_wins']} "
          f"mcts_white={r['mcts_white_wins']} ab_black={r['ab_black_wins']} "
          f"ab_white={r['ab_white_wins']}")
    print(f"MCTS win rate over decided games: {winrate:.1%}")

    if args.min_winrate is not None:
        if not decided:
            raise AssertionError("no decided games; cannot judge the win rate")
        if winrate < args.min_winrate:
            raise AssertionError(
                f"MCTS win rate {winrate:.1%} < required {args.min_winrate:.1%}")
        print("OK: head-to-head gate passed")
    else:
        print("OK: head-to-head gate passed (report-only)")


if __name__ == "__main__":
    main()
