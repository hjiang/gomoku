#include "core/Board.hpp"
#include "core/WinDetector.hpp"

#include <catch2/catch_test_macros.hpp>

using namespace gomoku;

namespace {

// Places stones in a row from (row, startCol) to (row, endCol) inclusive.
void placeHorizontal(Board& b, Player p, int row, int startCol, int endCol) {
  for (int c = startCol; c <= endCol; ++c) {
    b.place({Position{row, c}, p});
  }
}

void placeVertical(Board& b, Player p, int col, int startRow, int endRow) {
  for (int r = startRow; r <= endRow; ++r) {
    b.place({Position{r, col}, p});
  }
}

void placeDiagonal(Board& b, Player p, int startRow, int startCol, int length) {
  for (int i = 0; i < length; ++i) {
    b.place({Position{startRow + i, startCol + i}, p});
  }
}

void placeAntiDiagonal(Board& b, Player p, int startRow, int startCol, int length) {
  for (int i = 0; i < length; ++i) {
    b.place({Position{startRow + i, startCol - i}, p});
  }
}

}  // namespace

TEST_CASE("horizontal five-in-a-row wins", "[windetector]") {
  Board b;
  placeHorizontal(b, Player::Black, 7, 3, 7);
  REQUIRE(WinDetector::winnerOf(b, Position{7, 5}) == Player::Black);
}

TEST_CASE("vertical five-in-a-row wins", "[windetector]") {
  Board b;
  placeVertical(b, Player::White, 4, 2, 6);
  REQUIRE(WinDetector::winnerOf(b, Position{4, 4}) == Player::White);
}

TEST_CASE("main-diagonal five-in-a-row wins", "[windetector]") {
  Board b;
  placeDiagonal(b, Player::Black, 3, 5, 5);
  REQUIRE(WinDetector::winnerOf(b, Position{5, 7}) == Player::Black);
}

TEST_CASE("anti-diagonal five-in-a-row wins", "[windetector]") {
  Board b;
  placeAntiDiagonal(b, Player::White, 3, 9, 5);
  REQUIRE(WinDetector::winnerOf(b, Position{5, 7}) == Player::White);
}

TEST_CASE("overline (six in a row) still wins", "[windetector]") {
  Board b;
  placeHorizontal(b, Player::Black, 7, 0, 5);
  REQUIRE(WinDetector::winnerOf(b, Position{7, 2}) == Player::Black);
}

TEST_CASE("four in a row is not a win", "[windetector]") {
  Board b;
  placeHorizontal(b, Player::Black, 7, 3, 6);
  REQUIRE(WinDetector::winnerOf(b, Position{7, 4}) == Player::None);
}

TEST_CASE("broken line (gap) is not a win", "[windetector]") {
  Board b;
  placeHorizontal(b, Player::Black, 7, 3, 6);   // four stones
  b.place({Position{7, 8}, Player::Black});     // gap at col 7
  REQUIRE(WinDetector::winnerOf(b, Position{7, 4}) == Player::None);
}

TEST_CASE("win at the board edge", "[windetector]") {
  Board b;
  placeHorizontal(b, Player::White, 0, 0, 4);   // against top edge
  REQUIRE(WinDetector::winnerOf(b, Position{0, 2}) == Player::White);
}

TEST_CASE("opponent stones break the run", "[windetector]") {
  Board b;
  placeHorizontal(b, Player::Black, 7, 3, 6);
  b.place({Position{7, 7}, Player::White});     // blocks the run
  b.place({Position{7, 8}, Player::Black});
  REQUIRE(WinDetector::winnerOf(b, Position{7, 5}) == Player::None);
}

TEST_CASE("findWinningLine returns the five cells", "[windetector]") {
  Board b;
  placeDiagonal(b, Player::Black, 2, 2, 5);
  const auto line = WinDetector::findWinningLine(b, Position{4, 4});
  REQUIRE(line.has_value());
  REQUIRE((*line)[0] == Position{2, 2});
  REQUIRE((*line)[4] == Position{6, 6});
}

TEST_CASE("findWinningLine returns nullopt without a win", "[windetector]") {
  Board b;
  placeHorizontal(b, Player::Black, 7, 3, 6);
  REQUIRE_FALSE(WinDetector::findWinningLine(b, Position{7, 4}).has_value());
}

TEST_CASE("anyWinner detects a win on the whole board", "[windetector]") {
  Board b;
  placeVertical(b, Player::White, 10, 5, 9);
  REQUIRE(WinDetector::anyWinner(b) == Player::White);
}

TEST_CASE("anyWinner is None without a win", "[windetector]") {
  Board b;
  placeHorizontal(b, Player::Black, 7, 3, 6);   // only four
  placeHorizontal(b, Player::White, 8, 3, 6);
  REQUIRE(WinDetector::anyWinner(b) == Player::None);
}
