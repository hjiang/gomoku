#pragma once

#include "core/types.hpp"

#include <array>
#include <cstddef>
#include <optional>
#include <vector>

namespace gomoku {

// Fixed kSize x kSize board with move history (for undo and last-move lookup).
class Board {
 public:
  Board() = default;

  // Stone at `pos` (Player::None when empty). Pre: inBounds(pos).
  [[nodiscard]] Player at(Position pos) const;

  [[nodiscard]] bool inBounds(Position pos) const;

  // True when pos is in bounds and empty; false when out of bounds.
  [[nodiscard]] bool isEmpty(Position pos) const;

  [[nodiscard]] bool isFull() const;

  [[nodiscard]] int moveCount() const { return static_cast<int>(history_.size()); }

  [[nodiscard]] std::optional<Move> lastMove() const;

  const std::vector<Move>& history() const { return history_; }

  // Pre: inBounds(move.pos) && isEmpty(move.pos) && move.player != None.
  // Post: at(move.pos) == move.player && moveCount() == old + 1.
  void place(Move move);

  // Pre: moveCount() > 0. Post: board restored to the state before the last move.
  void undo();

 private:
  [[nodiscard]] std::size_t index(Position pos) const {
    return static_cast<std::size_t>(pos.row) * kSize + static_cast<std::size_t>(pos.col);
  }

  std::array<Player, static_cast<std::size_t>(kSize) * kSize> cells_{};
  std::vector<Move> history_;
};

}  // namespace gomoku
