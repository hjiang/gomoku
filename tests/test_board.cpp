#include "core/Board.hpp"

#include <catch2/catch_test_macros.hpp>

using namespace gomoku;

TEST_CASE("empty board is not full", "[board]") {
  Board b;
  REQUIRE_FALSE(b.isFull());
  REQUIRE(b.moveCount() == 0);
  REQUIRE_FALSE(b.lastMove().has_value());
}

TEST_CASE("place and read back a stone", "[board]") {
  Board b;
  b.place({Position{3, 4}, Player::Black});
  REQUIRE(b.at(Position{3, 4}) == Player::Black);
  REQUIRE(b.moveCount() == 1);
  REQUIRE(b.lastMove()->pos == Position{3, 4});
  REQUIRE(b.lastMove()->player == Player::Black);
}

TEST_CASE("out of bounds positions are not empty", "[board]") {
  Board b;
  REQUIRE_FALSE(b.inBounds(Position{-1, 0}));
  REQUIRE_FALSE(b.inBounds(Position{0, kSize}));
  REQUIRE_FALSE(b.inBounds(Position{kSize, 0}));
  REQUIRE_FALSE(b.isEmpty(Position{-1, -1}));
}

TEST_CASE("undo restores the previous state", "[board]") {
  Board b;
  b.place({Position{1, 1}, Player::Black});
  b.place({Position{2, 2}, Player::White});
  REQUIRE(b.moveCount() == 2);
  b.undo();
  REQUIRE(b.moveCount() == 1);
  REQUIRE(b.at(Position{2, 2}) == Player::None);
  REQUIRE(b.at(Position{1, 1}) == Player::Black);
  b.undo();
  REQUIRE(b.moveCount() == 0);
  REQUIRE(b.at(Position{1, 1}) == Player::None);
}

TEST_CASE("occupied cell cannot be replayed", "[board]") {
  Board b;
  b.place({Position{5, 5}, Player::Black});
  REQUIRE_FALSE(b.isEmpty(Position{5, 5}));
}

TEST_CASE("board is full only after 225 stones", "[board]") {
  Board b;
  // Deterministic fill that never makes five in a row: colour cell (r,c) by
  // (2r + c) mod 5. Every line changes that value by a unit coprime to 5, so
  // no five consecutive cells share a colour.
  for (int r = 0; r < kSize; ++r) {
    for (int c = 0; c < kSize; ++c) {
      b.place({Position{r, c}, ((2 * r + c) % 5 < 2) ? Player::Black : Player::White});
    }
  }
  REQUIRE(b.isFull());
}
