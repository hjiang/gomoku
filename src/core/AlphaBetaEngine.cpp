#include "core/AlphaBetaEngine.hpp"

#include "core/Evaluator.hpp"
#include "core/WinDetector.hpp"

#include <algorithm>
#include <chrono>
#include <cassert>
#include <cstdlib>
#include <limits>
#include <optional>
#include <random>
#include <utility>
#include <vector>

namespace gomoku {
namespace {

constexpr int kInfinity = std::numeric_limits<int>::max() / 4;
// Larger than any sum of evaluator scores (max |eval| << 10^7).
constexpr int kWinScore = 10000000;

struct SearchContext {
  const SearchParams& params;
  std::chrono::steady_clock::time_point deadline;
  int nodes = 0;

  [[nodiscard]] bool outOfTime() const {
    return std::chrono::steady_clock::now() >= deadline;
  }
};

// Empty cells within Chebyshev distance 2 of an occupied cell. An empty board
// is handled by the caller before reaching this point.
std::vector<Position> generateCandidates(const Board& board) {
  std::vector<Position> candidates;
  for (int r = 0; r < kSize; ++r) {
    for (int c = 0; c < kSize; ++c) {
      const Position pos{r, c};
      if (!board.isEmpty(pos)) {
        continue;
      }
      bool nearStone = false;
      for (int dr = -2; dr <= 2 && !nearStone; ++dr) {
        for (int dc = -2; dc <= 2 && !nearStone; ++dc) {
          if (dr == 0 && dc == 0) {
            continue;
          }
          const Position n{r + dr, c + dc};
          if (board.inBounds(n) && board.at(n) != Player::None) {
            nearStone = true;
          }
        }
      }
      if (nearStone) {
        candidates.push_back(pos);
      }
    }
  }
  // Defensive fallback: any empty cell.
  if (candidates.empty()) {
    for (int r = 0; r < kSize; ++r) {
      for (int c = 0; c < kSize; ++c) {
        if (board.isEmpty(Position{r, c})) {
          candidates.push_back(Position{r, c});
        }
      }
    }
  }
  return candidates;
}

// Orders candidates deterministically: immediate wins first, then cells close
// to the last move, then row-major order (stable).
void orderMoves(const Board& board, std::vector<Position>& moves, Player side) {
  const std::optional<Position> last =
      board.lastMove() ? std::optional<Position>(board.lastMove()->pos) : std::nullopt;

  auto rank = [&](Position pos) {
    int score = 0;
    Board probe = board;
    probe.place({pos, side});
    if (WinDetector::winnerOf(probe, pos) == side) {
      score -= 1'000'000;  // wins sort first
    }
    if (last) {
      const int d = std::max(std::abs(pos.row - last->row), std::abs(pos.col - last->col));
      score += d * 1000;
    }
    return score;
  };

  std::stable_sort(moves.begin(), moves.end(),
                   [&](Position a, Position b) { return rank(a) < rank(b); });
}

// Negamax with alpha-beta. `side` is the player to move at this node; `depth`
// is the number of plies still to search. Returns the value from `side`'s
// perspective. Mutates `board` (caller restores via place/undo pairing).
int negamax(Board& board, Player side, int depth, int alpha, int beta, SearchContext& ctx) {
  if (ctx.outOfTime()) {
    return 0;
  }
  if (board.isFull()) {
    return 0;
  }
  if (depth == 0) {
    const int eval = Evaluator::staticScore(board);
    return side == Player::Black ? eval : -eval;
  }

  std::vector<Position> moves = generateCandidates(board);
  orderMoves(board, moves, side);

  int best = -kInfinity;
  for (const Position pos : moves) {
    if (ctx.outOfTime()) {
      break;
    }
    ++ctx.nodes;

    const Move move{pos, side};
    board.place(move);
    int value = 0;
    if (WinDetector::winnerOf(board, pos) == side) {
      // Depth bonus prefers faster wins and slower losses.
      value = kWinScore + depth;
    } else {
      value = -negamax(board, opponent(side), depth - 1, -beta, -alpha, ctx);
    }
    board.undo();

    if (value > best) {
      best = value;
    }
    if (best > alpha) {
      alpha = best;
    }
    if (alpha >= beta) {
      break;
    }
  }
  return best == -kInfinity ? 0 : best;
}

}  // namespace

Move AlphaBetaEngine::findBestMove(const Board& board, Player player, const SearchParams& params) {
  assert(WinDetector::anyWinner(board) == Player::None);
  assert(board.moveCount() < kSize * kSize);

  std::vector<Position> moves = generateCandidates(board);
  if (params.seed != 0) {
    std::mt19937 rng(params.seed);
    std::shuffle(moves.begin(), moves.end(), rng);
  }
  // Always order the root moves (immediate wins first, then near the last
  // move); the shuffle above only varies the order among equally-ranked moves.
  orderMoves(board, moves, player);

  assert(!moves.empty());
  Move best{moves.front(), player};

  SearchContext ctx{params,
                    std::chrono::steady_clock::now() + std::chrono::milliseconds(params.timeBudgetMs)};

  for (int depth = 1; depth <= params.maxDepth; ++depth) {
    if (ctx.outOfTime()) {
      break;
    }
    int alpha = -kInfinity;
    int beta = kInfinity;
    Move bestThisDepth{};
    int bestScore = -kInfinity;

    for (const Position pos : moves) {
      if (ctx.outOfTime()) {
        break;
      }
      ++ctx.nodes;

      Board next = board;
      next.place({pos, player});
      int value = 0;
      if (WinDetector::winnerOf(next, pos) == player) {
        value = kWinScore + depth;
      } else {
        value = -negamax(next, opponent(player), depth - 1, -beta, -alpha, ctx);
      }
      if (value > bestScore) {
        bestScore = value;
        bestThisDepth = Move{pos, player};
      }
      if (bestScore > alpha) {
        alpha = bestScore;
      }
      if (alpha >= beta) {
        break;
      }
    }

    if (ctx.outOfTime()) {
      // This depth is incomplete; keep the move from the last completed depth.
      break;
    }
    best = bestThisDepth;
    if (bestScore >= kWinScore) {
      break;  // forced win found; deeper search cannot improve it
    }
  }

  return best;
}

}  // namespace gomoku
