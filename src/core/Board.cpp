#include "core/Board.hpp"

#include <cassert>

namespace gomoku {

Player Board::at(Position pos) const {
  assert(inBounds(pos));
  return cells_[index(pos)];
}

bool Board::inBounds(Position pos) const {
  return pos.row >= 0 && pos.row < kSize && pos.col >= 0 && pos.col < kSize;
}

bool Board::isEmpty(Position pos) const {
  return inBounds(pos) && at(pos) == Player::None;
}

bool Board::isFull() const {
  return moveCount() == kSize * kSize;
}

std::optional<Move> Board::lastMove() const {
  if (history_.empty()) {
    return std::nullopt;
  }
  return history_.back();
}

void Board::place(Move move) {
  assert(inBounds(move.pos));
  assert(move.player != Player::None);
  assert(isEmpty(move.pos));
  cells_[index(move.pos)] = move.player;
  history_.push_back(move);
}

void Board::undo() {
  assert(!history_.empty());
  const Move last = history_.back();
  history_.pop_back();
  cells_[index(last.pos)] = Player::None;
}

}  // namespace gomoku
