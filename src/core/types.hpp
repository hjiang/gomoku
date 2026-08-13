#pragma once

#include <cstdint>

namespace gomoku {

// Fixed board size (freestyle 15x15). A compile-time constant so the board
// could be enlarged (e.g. 19x19) with minimal change.
inline constexpr int kSize = 15;

enum class Player : std::uint8_t {
  None = 0,
  Black = 1,
  White = 2,
};

struct Position {
  int row = 0;
  int col = 0;

  friend constexpr auto operator<=>(const Position&, const Position&) = default;
};

struct Move {
  Position pos{};
  Player player = Player::None;

  friend constexpr auto operator<=>(const Move&, const Move&) = default;
};

// The other player; precondition: p is Black or White.
inline constexpr Player opponent(Player p) noexcept {
  return p == Player::Black ? Player::White : Player::Black;
}

}  // namespace gomoku
