#include "core/BoardEncoder.hpp"

namespace gomoku {

std::array<float, BoardEncoder::kInputSize> BoardEncoder::encode(const Board& board,
                                                                 Player toMove) {
  std::array<float, kInputSize> out{};
  const int plane = kSize * kSize;

  for (int r = 0; r < kSize; ++r) {
    for (int c = 0; c < kSize; ++c) {
      const Player stone = board.at(Position{r, c});
      const int idx = r * kSize + c;
      if (stone == toMove) {
        out[idx] = 1.0f;  // plane 0
      } else if (stone == opponent(toMove)) {
        out[plane + idx] = 1.0f;  // plane 1
      }
    }
  }

  for (int i = 0; i < plane; ++i) {
    out[2 * plane + i] = 1.0f;  // plane 2: to-move fill
  }

  const std::optional<Move> last = board.lastMove();
  if (last) {
    out[3 * plane + last->pos.row * kSize + last->pos.col] = 1.0f;  // plane 3
  }
  return out;
}

}  // namespace gomoku
