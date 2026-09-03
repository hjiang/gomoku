#pragma once

#include "core/Board.hpp"

#include <chrono>
#include <cstdint>

namespace gomoku {

// Which AI search engine to run.
enum class EngineKind { AlphaBeta, Mcts };

struct SearchParams {
  int maxDepth = 4;           // plies of search at the root (AlphaBeta)
  int timeBudgetMs = 2000;    // soft deadline; the search stops after it
  std::uint32_t seed = 0;     // nonzero: deterministic variety in move ordering
  EngineKind engine = EngineKind::AlphaBeta;  // which search engine to run
  int mctsSimulations = 1000; // per-move simulation budget (Mcts)
  int openingJitter = 0;      // first N plies of a bootstrap game are uniform-random (center region)
  int openingRadius = 2;      // half-width of the random-opening region (center 5x5)
};

// Facade: dispatches to the selected engine behind a single board -> move
// contract. See AlphaBetaEngine and MctsEngine.
class SearchEngine {
 public:
  // Best move for `player`. Pre: board has at least one empty cell and no
  // winner. Post: the returned move is legal and empty.
  [[nodiscard]] static Move findBestMove(const Board& board, Player player, const SearchParams& params);

  // Difficulty presets: 0 = Easy, 1 = Medium, 2 = Hard.
  [[nodiscard]] static SearchParams difficulty(int level);
};

}  // namespace gomoku
