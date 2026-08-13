#pragma once

#include "core/Board.hpp"
#include "core/SearchEngine.hpp"

#include <array>
#include <cstdint>
#include <span>

namespace gomoku {

// Monte-Carlo tree search (PUCT) guided by a trained policy/value network
// (AlphaZero-style). Requires a trained model: until weights are loaded,
// findBestMove throws (no heuristic fallback).
class MctsEngine {
 public:
  // Whether a trained model is currently loaded.
  [[nodiscard]] static bool isModelAvailable();

  // Parses a .gnn weight file from `bytes` and builds the network. Returns
  // true on success; on failure the previously loaded model (if any) is left
  // unchanged. Loading happens once at startup, before any search runs.
  [[nodiscard]] static bool loadModel(std::span<const std::uint8_t> bytes);

  // Clears any loaded model (mainly for tests that need the "no model"
  // baseline back; harmless to call when nothing is loaded).
  static void unloadModel();

  // Pre: isModelAvailable(). Throws std::runtime_error otherwise.
  [[nodiscard]] static Move findBestMove(const Board& board, Player player, const SearchParams& params);

  // Result of a self-play search: the chosen move plus the root visit-count
  // policy target (normalized, row-major; unvisited cells are 0).
  struct SelfPlayResult {
    Move move;
    std::array<float, kSize * kSize> policy{};
  };

  // AlphaZero-style self-play move selection: PUCT with root Dirichlet noise
  // (see PLAN-stage3-training.md) driven by params.seed. Pre: isModelAvailable()
  // and the board has no winner and is not full. Post: policy sums to 1 and
  // move is the argmax-visit-count legal move (row-major tie-break).
  [[nodiscard]] static SelfPlayResult selfPlay(const Board& board, Player player,
                                               const SearchParams& params);
};

}  // namespace gomoku
