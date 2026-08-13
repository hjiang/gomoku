#pragma once

#include "core/Board.hpp"

namespace gomoku {

// Static pattern-based evaluation of a board position (Qt-free, pure).
class Evaluator {
 public:
  // Score of `player`'s stones alone: higher is better for `player`.
  [[nodiscard]] static int score(const Board& board, Player player);

  // Black's score minus White's score.
  [[nodiscard]] static int staticScore(const Board& board);
};

}  // namespace gomoku
