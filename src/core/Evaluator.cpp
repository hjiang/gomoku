#include "core/Evaluator.hpp"

#include "core/PatternTable.hpp"

#include <array>
#include <cassert>
#include <utility>

namespace gomoku {
namespace {

constexpr std::array<std::pair<int, int>, 4> kDirections = {{
    {0, 1},
    {1, 0},
    {1, 1},
    {1, -1},
}};

// Score of the length-kWindowSize window whose first cell is `start`, or 0
// when the window crosses the board edge or contains an opponent stone.
int windowScoreAt(const Board& board, Player player, Position start, int dr, int dc) {
  int count = 0;
  for (int i = 0; i < PatternTable::kWindowSize; ++i) {
    const Position pos{start.row + dr * i, start.col + dc * i};
    if (!board.inBounds(pos)) {
      return 0;
    }
    const Player p = board.at(pos);
    if (p == player) {
      ++count;
    } else if (p != Player::None) {
      return 0;
    }
  }
  return PatternTable::windowScore(count);
}

}  // namespace

int Evaluator::score(const Board& board, Player player) {
  assert(player == Player::Black || player == Player::White);

  int total = 0;
  for (const auto& [dr, dc] : kDirections) {
    for (int r = 0; r < kSize; ++r) {
      for (int c = 0; c < kSize; ++c) {
        total += windowScoreAt(board, player, Position{r, c}, dr, dc);
      }
    }
  }
  return total;
}

int Evaluator::staticScore(const Board& board) {
  return score(board, Player::Black) - score(board, Player::White);
}

}  // namespace gomoku
