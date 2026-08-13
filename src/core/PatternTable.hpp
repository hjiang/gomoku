#pragma once

#include <array>

namespace gomoku {

// Scores for the window-based pattern evaluator. A window of kWindowSize
// consecutive cells containing exactly `n` stones of one colour and no
// opponent stone scores windowScore(n); windows touched by an opponent stone
// score zero. Values grow steeply with n so the heuristic is monotone:
// five > four > three > two > one.
class PatternTable {
 public:
  static constexpr int kWindowSize = 6;

  // Lazy-initialized per-window score table (index 0 is unused).
  [[nodiscard]] static const std::array<int, kWindowSize + 1>& windowScores();

  [[nodiscard]] static int windowScore(int count);
};

}  // namespace gomoku
