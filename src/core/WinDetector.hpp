#pragma once

#include "core/Board.hpp"

#include <array>
#include <optional>

namespace gomoku {

// Pure win/draw detection. Freestyle gomoku: five OR MORE in a line wins.
class WinDetector {
 public:
  // Winner of the line passing through `last`, or Player::None.
  // Pre: inBounds(last) && at(last) != None.
  static Player winnerOf(const Board& board, Position last);

  // The 5 cells of the winning line through `last` (for UI highlighting),
  // or nullopt when there is no win. Pre: inBounds(last) && at(last) != None.
  static std::optional<std::array<Position, 5>> findWinningLine(const Board& board, Position last);

  // Any winner on the whole board (used for draw detection and engine leaves).
  static Player anyWinner(const Board& board);
};

}  // namespace gomoku
