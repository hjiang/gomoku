#include "core/MctsEngine.hpp"
#include "core/SearchEngine.hpp"
#include "core/WinDetector.hpp"

#include <catch2/catch_test_macros.hpp>

#include <stdexcept>

using namespace gomoku;

namespace {

// White has four at (7, 3..6) with the right end capped by Black at (7,7);
// the only move that stops White from winning at (7,2) is Black's (7,2).
Board openFourThreat() {
  Board b;
  for (int c = 3; c <= 6; ++c) {
    b.place({Position{7, c}, Player::White});
  }
  b.place({Position{7, 7}, Player::Black});  // caps the right end
  b.place({Position{5, 5}, Player::Black});  // a couple of Black stones
  b.place({Position{9, 9}, Player::Black});
  return b;
}

}  // namespace

TEST_CASE("default search params use the alpha-beta engine", "[engine][dispatch]") {
  const SearchParams params;
  REQUIRE(params.engine == EngineKind::AlphaBeta);
}

TEST_CASE("alpha-beta engine blocks an opponent open four", "[engine][dispatch]") {
  Board b = openFourThreat();
  SearchParams params = SearchEngine::difficulty(1);
  params.engine = EngineKind::AlphaBeta;
  const Move m = SearchEngine::findBestMove(b, Player::Black, params);
  REQUIRE(m.pos == Position{7, 2});
}

TEST_CASE("MCTS engine without a model throws", "[engine][dispatch]") {
  MctsEngine::unloadModel();  // decouple from model state left by other test files
  Board b = openFourThreat();
  SearchParams params = SearchEngine::difficulty(1);
  params.engine = EngineKind::Mcts;
  REQUIRE_THROWS_AS(SearchEngine::findBestMove(b, Player::Black, params), std::runtime_error);
}

TEST_CASE("MCTS engine reports no model available before Stage 2", "[engine][dispatch]") {
  MctsEngine::unloadModel();  // decouple from model state left by other test files
  REQUIRE_FALSE(MctsEngine::isModelAvailable());
}

TEST_CASE("difficulty presets scale both engines", "[engine][dispatch]") {
  const SearchParams easy = SearchEngine::difficulty(0);
  const SearchParams hard = SearchEngine::difficulty(2);
  REQUIRE(easy.maxDepth < hard.maxDepth);
  REQUIRE(easy.mctsSimulations < hard.mctsSimulations);
}
