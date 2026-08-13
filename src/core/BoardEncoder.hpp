#pragma once

#include "core/Board.hpp"

#include <array>

namespace gomoku {

// Encodes a board into the neural network's input tensor: 4 binary planes of
// kSize x kSize, channels-first.
//   plane 0: stones of `toMove`
//   plane 1: stones of the opponent
//   plane 2: constant 1.0 (player-to-move fill)
//   plane 3: last-move marker (1.0 at the last move, else 0)
class BoardEncoder {
 public:
  static constexpr int kPlanes = 4;
  static constexpr int kInputSize = kPlanes * kSize * kSize;

  // Pre: none. Post: every element is 0.0f or 1.0f.
  [[nodiscard]] static std::array<float, kInputSize> encode(const Board& board, Player toMove);
};

}  // namespace gomoku
