#include "core/SearchEngine.hpp"

#include "core/AlphaBetaEngine.hpp"
#include "core/MctsEngine.hpp"

#include <cassert>

namespace gomoku {

SearchParams SearchEngine::difficulty(int level) {
  SearchParams params;
  switch (level) {
    case 0:
      params.maxDepth = 2;
      params.timeBudgetMs = 500;
      params.mctsSimulations = 400;
      break;
    case 1:
      params.maxDepth = 4;
      params.timeBudgetMs = 1500;
      params.mctsSimulations = 800;
      break;
    default:
      params.maxDepth = 6;
      params.timeBudgetMs = 2000;
      params.mctsSimulations = 1600;
      break;
  }
  return params;
}

Move SearchEngine::findBestMove(const Board& board, Player player, const SearchParams& params) {
  switch (params.engine) {
    case EngineKind::AlphaBeta:
      return AlphaBetaEngine::findBestMove(board, player, params);
    case EngineKind::Mcts:
      return MctsEngine::findBestMove(board, player, params);
  }
  assert(false && "unreachable engine kind");
  return Move{};
}

}  // namespace gomoku
