#include "core/Board.hpp"
#include "core/Evaluator.hpp"

#include <catch2/catch_test_macros.hpp>

using namespace gomoku;

namespace {

// Builds a board where `player` has a line of `count` consecutive stones at
// (row, startCol..startCol+count-1); the rest of the board stays empty.
Board lineOf(Player player, int count) {
  Board b;
  for (int c = 0; c < count; ++c) {
    b.place({Position{7, 5 + c}, player});
  }
  return b;
}

Board swapped(const Board& b) {
  Board out;
  for (int r = 0; r < kSize; ++r) {
    for (int c = 0; c < kSize; ++c) {
      const Player p = b.at(Position{r, c});
      if (p == Player::Black) {
        out.place({Position{r, c}, Player::White});
      } else if (p == Player::White) {
        out.place({Position{r, c}, Player::Black});
      }
    }
  }
  return out;
}

}  // namespace

TEST_CASE("empty board scores zero", "[evaluator]") {
  Board b;
  REQUIRE(Evaluator::score(b, Player::Black) == 0);
  REQUIRE(Evaluator::score(b, Player::White) == 0);
  REQUIRE(Evaluator::staticScore(b) == 0);
}

TEST_CASE("longer lines score higher (monotonic)", "[evaluator]") {
  int prev = 0;
  for (int count = 1; count <= 5; ++count) {
    const int s = Evaluator::score(lineOf(Player::Black, count), Player::Black);
    REQUIRE(s > prev);
    prev = s;
  }
}

TEST_CASE("open formations score more than blocked ones", "[evaluator]") {
  // Open four: _XXXX_ on an empty board.
  Board open;
  for (int c = 3; c <= 6; ++c) {
    open.place({Position{7, c}, Player::Black});
  }
  // Blocked four: opponent caps one end.
  Board blocked = open;
  blocked.place({Position{7, 2}, Player::White});

  REQUIRE(Evaluator::score(open, Player::Black) > Evaluator::score(blocked, Player::Black));
}

TEST_CASE("symmetry: score(b, Black) == score(swapped b, White)", "[evaluator]") {
  Board b = lineOf(Player::Black, 3);
  b.place({Position{3, 3}, Player::White});
  b.place({Position{11, 11}, Player::White});

  REQUIRE(Evaluator::score(b, Player::Black) == Evaluator::score(swapped(b), Player::White));
}

TEST_CASE("static score flips sign when colours are swapped", "[evaluator]") {
  Board b = lineOf(Player::Black, 4);
  REQUIRE(Evaluator::staticScore(b) == -Evaluator::staticScore(swapped(b)));
}
