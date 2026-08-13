#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace gomoku {

// Parsed weight tensors for the neural network, channels-first float32.
// Loaded from the .gnn byte format (see docs/plans/PLAN-neural-mcts.md and
// Weights.cpp). Pure and hermetic: `parse` only reads from memory.
struct Weights {
  int numBlocks = 0;
  int channels = 0;

  // Input projection: conv(4 -> C) + BN + ReLU.
  std::vector<float> inputConvW;  // (C, 4, 3, 3)
  std::vector<float> inputConvB;  // (C)
  std::vector<float> inputBnGamma;
  std::vector<float> inputBnBeta;
  std::vector<float> inputBnMean;
  std::vector<float> inputBnVar;  // (C) each

  // One residual block: ReLU(BN(conv1(x))) then BN(conv2(h)) then ReLU(x + h).
  struct Block {
    std::vector<float> conv1W;  // (C, C, 3, 3)
    std::vector<float> conv1B;  // (C)
    std::vector<float> bn1Gamma;
    std::vector<float> bn1Beta;
    std::vector<float> bn1Mean;
    std::vector<float> bn1Var;
    std::vector<float> conv2W;  // (C, C, 3, 3)
    std::vector<float> conv2B;  // (C)
    std::vector<float> bn2Gamma;
    std::vector<float> bn2Beta;
    std::vector<float> bn2Mean;
    std::vector<float> bn2Var;
  };
  std::vector<Block> blocks;  // numBlocks entries

  // Policy head: conv(2) -> flatten(450) -> linear(450 -> 225).
  std::vector<float> policyConvW;  // (2, C, 3, 3)
  std::vector<float> policyConvB;  // (2)
  std::vector<float> policyFcW;    // (225, 450)
  std::vector<float> policyFcB;    // (225)

  // Value head: conv(1) -> flatten(225) -> linear(225 -> 256) -> ReLU ->
  // linear(256 -> 1) -> tanh.
  std::vector<float> valueConvW;  // (1, C, 3, 3)
  std::vector<float> valueConvB;  // (1)
  std::vector<float> valueFc1W;   // (256, 225)
  std::vector<float> valueFc1B;   // (256)
  std::vector<float> valueFc2W;   // (1, 256)
  std::vector<float> valueFc2B;   // (1)

  // Parses the .gnn format from a byte buffer. Returns nullopt on any
  // malformed input: bad magic, unsupported version, numBlocks/channels < 1,
  // or a buffer whose length does not exactly match the layout. Reads f32 as
  // little-endian (host must be little-endian; asserted at compile time).
  [[nodiscard]] static std::optional<Weights> parse(std::span<const std::uint8_t> bytes);
};

}  // namespace gomoku
