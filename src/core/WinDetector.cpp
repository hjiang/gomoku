#include "core/WinDetector.hpp"

#include <array>
#include <cassert>
#include <utility>

namespace gomoku {
namespace {

constexpr std::array<std::pair<int, int>, 4> kDirections = {{
    {0, 1},   // horizontal
    {1, 0},   // vertical
    {1, 1},   // main diagonal
    {1, -1},  // anti diagonal
}};

// Consecutive stones equal to `player` counting from `start` and moving by
// (dr, dc), including `start` itself.
int runLength(const Board& board, Position start, int dr, int dc, Player player) {
  int count = 0;
  Position cur = start;
  while (board.inBounds(cur) && board.at(cur) == player) {
    ++count;
    cur = Position{cur.row + dr, cur.col + dc};
  }
  return count;
}

}  // namespace

Player WinDetector::winnerOf(const Board& board, Position last) {
  assert(board.inBounds(last));
  assert(board.at(last) != Player::None);

  const Player player = board.at(last);
  for (const auto& [dr, dc] : kDirections) {
    const Position before = Position{last.row - dr, last.col - dc};
    const int backward = runLength(board, before, -dr, -dc, player);
    const int forward = runLength(board, last, dr, dc, player);
    if (backward + forward >= 5) {
      return player;
    }
  }
  return Player::None;
}

std::optional<std::array<Position, 5>> WinDetector::findWinningLine(const Board& board, Position last) {
  assert(board.inBounds(last));
  assert(board.at(last) != Player::None);

  const Player player = board.at(last);
  for (const auto& [dr, dc] : kDirections) {
    const int forward = runLength(board, last, dr, dc, player);
    int backward = 0;
    Position cur = Position{last.row - dr, last.col - dc};
    while (board.inBounds(cur) && board.at(cur) == player) {
      ++backward;
      cur = Position{cur.row - dr, cur.col - dc};
    }
    if (backward + forward < 5) {
      continue;
    }
    // The run starts `backward` cells behind `last`.
    const Position start = Position{last.row - dr * backward, last.col - dc * backward};
    std::array<Position, 5> line{};
    for (int i = 0; i < 5; ++i) {
      line[static_cast<std::size_t>(i)] = Position{start.row + dr * i, start.col + dc * i};
    }
    return line;
  }
  return std::nullopt;
}

Player WinDetector::anyWinner(const Board& board) {
  for (int r = 0; r < kSize; ++r) {
    for (int c = 0; c < kSize; ++c) {
      const Position pos{r, c};
      if (board.at(pos) != Player::None && winnerOf(board, pos) != Player::None) {
        return board.at(pos);
      }
    }
  }
  return Player::None;
}

}  // namespace gomoku
