#pragma once

#include "core/Board.hpp"
#include "core/SearchEngine.hpp"

namespace gomoku {

// Classic search: negamax + alpha-beta + iterative deepening over candidate
// moves near stones, evaluated by the pattern heuristic.
class AlphaBetaEngine {
 public:
  // Best move for `player`. Pre: board has at least one empty cell and no
  // winner. Post: the returned move is legal and empty.
  [[nodiscard]] static Move findBestMove(const Board& board, Player player, const SearchParams& params);
};

}  // namespace gomoku
