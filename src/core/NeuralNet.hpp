#pragma once

#include "core/Weights.hpp"
#include "core/types.hpp"

#include <array>
#include <span>

namespace gomoku {

// Forward-only inference for the gomoku policy/value network. Pure, Qt-free,
// float32, no external dependencies. See docs/plans/PLAN-neural-mcts.md for
// the architecture and .gnn format.
class NeuralNet {
 public:
  struct Output {
    // Raw policy logits for all kSize*kSize cells (no masking, no softmax —
    // the caller masks illegal moves). Value in [-1, 1] from the perspective
    // of the player to move.
    std::array<float, kSize * kSize> policyLogits{};
    float value = 0.0f;
  };

  // Pre: weights was produced by Weights::parse.
  explicit NeuralNet(Weights weights);

  // Pre: input.size() == BoardEncoder::kInputSize, channels-first.
  // Post: policyLogits holds the 225 logits; value is finite in [-1, 1].
  [[nodiscard]] Output evaluate(std::span<const float> input) const;

 private:
  Weights weights_;
};

}  // namespace gomoku
