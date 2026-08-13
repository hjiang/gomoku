#pragma once

#include "core/Board.hpp"

#include <chrono>
#include <cstdint>

namespace gomoku {

struct SearchParams {
  int maxDepth = 4;          // plies of search at the root
  int timeBudgetMs = 2000;   // soft deadline; the search stops after it
  std::uint32_t seed = 0;    // nonzero: deterministic variety in move ordering
};

// Negamax + alpha-beta + iterative deepening over candidate moves near stones.
class SearchEngine {
 public:
  // Best move for `player`. Pre: board has at least one empty cell and no
  // winner. Post: the returned move is legal and empty.
  [[nodiscard]] static Move findBestMove(const Board& board, Player player, const SearchParams& params);

  // Difficulty presets: 0 = Easy, 1 = Medium, 2 = Hard.
  [[nodiscard]] static SearchParams difficulty(int level);
};

}  // namespace gomoku
