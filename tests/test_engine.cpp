#include "core/Board.hpp"
#include "core/SearchEngine.hpp"
#include "core/WinDetector.hpp"

#include <catch2/catch_test_macros.hpp>

#include <chrono>

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

// Black has four in a row at (7, 3..6); the last move completed it, so this
// board is already a win for the side that just moved. The engine's contract
// excludes this, so instead we build a board where the *player* can win now.
Board winInOneFor(Player p) {
  Board b;
  for (int c = 3; c <= 6; ++c) {
    b.place({Position{7, c}, p});
  }
  b.place({Position{1, 1}, opponent(p)});
  b.place({Position{13, 13}, opponent(p)});
  return b;  // p completes at (7,2) or (7,7)
}

}  // namespace

TEST_CASE("engine takes an immediate win", "[engine]") {
  Board b = winInOneFor(Player::Black);
  const Move m = SearchEngine::findBestMove(b, Player::Black, SearchEngine::difficulty(1));
  REQUIRE(b.isEmpty(m.pos));
  b.place(m);
  REQUIRE(WinDetector::winnerOf(b, m.pos) == Player::Black);
}

TEST_CASE("engine blocks an opponent open four", "[engine]") {
  Board b = openFourThreat();
  const Move m = SearchEngine::findBestMove(b, Player::Black, SearchEngine::difficulty(1));
  REQUIRE(m.pos == Position{7, 2});
}

TEST_CASE("engine returns a legal empty move on a near-full board", "[engine]") {
  Board b;
  for (int r = 0; r < kSize; ++r) {
    for (int c = 0; c < kSize; ++c) {
      if (r == 7 && c == 7) {
        continue;  // the single empty cell
      }
      b.place({Position{r, c}, ((2 * r + c) % 5 < 2) ? Player::Black : Player::White});
    }
  }
  REQUIRE(b.moveCount() == kSize * kSize - 1);
  const Move m = SearchEngine::findBestMove(b, Player::Black, SearchEngine::difficulty(0));
  REQUIRE(m.pos == Position{7, 7});
  REQUIRE(b.isEmpty(m.pos));
}

TEST_CASE("engine respects the time budget", "[engine]") {
  Board b;
  b.place({Position{7, 7}, Player::Black});
  b.place({Position{7, 8}, Player::White});
  b.place({Position{8, 7}, Player::Black});
  b.place({Position{8, 8}, Player::White});

  SearchParams params = SearchEngine::difficulty(2);
  params.timeBudgetMs = 100;   // tight budget on a midgame-ish board
  params.maxDepth = 8;

  const auto start = std::chrono::steady_clock::now();
  const Move m = SearchEngine::findBestMove(b, Player::Black, params);
  const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                           std::chrono::steady_clock::now() - start)
                           .count();

  REQUIRE(b.isEmpty(m.pos));
  REQUIRE(elapsed < 1000);  // budget (100 ms) plus generous scheduling slack
}

TEST_CASE("engine is deterministic for a fixed seed", "[engine]") {
  Board b = openFourThreat();
  SearchParams params = SearchEngine::difficulty(1);
  params.seed = 42;
  const Move a = SearchEngine::findBestMove(b, Player::Black, params);
  const Move c = SearchEngine::findBestMove(b, Player::Black, params);
  REQUIRE(b.isEmpty(a.pos));
  REQUIRE(a.pos == c.pos);
}

TEST_CASE("engine does not hang on an empty board", "[engine]") {
  Board b;
  const Move m = SearchEngine::findBestMove(b, Player::Black, SearchEngine::difficulty(2));
  REQUIRE(b.inBounds(m.pos));
  REQUIRE(b.isEmpty(m.pos));
}
