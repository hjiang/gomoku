#pragma once

#include "core/Board.hpp"
#include "core/SearchEngine.hpp"

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
};

}  // namespace gomoku
