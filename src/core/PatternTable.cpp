#include "core/PatternTable.hpp"

#include <array>

namespace gomoku {

const std::array<int, PatternTable::kWindowSize + 1>& PatternTable::windowScores() {
  static const std::array<int, kWindowSize + 1> scores = {0, 1, 10, 100, 1000, 10000, 100000};
  return scores;
}

int PatternTable::windowScore(int count) {
  return windowScores()[static_cast<std::size_t>(count)];
}

}  // namespace gomoku
