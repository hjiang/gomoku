#include "core/NeuralNet.hpp"

#include "core/TensorOps.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <vector>

namespace gomoku {

NeuralNet::NeuralNet(Weights weights) : weights_(std::move(weights)) {}

NeuralNet::Output NeuralNet::evaluate(std::span<const float> input) const {
  assert(input.size() == static_cast<std::size_t>(4) * kSize * kSize);

  const int C = weights_.channels;
  const int N = weights_.numBlocks;
  const int HW = kSize * kSize;

  // Input projection: conv(4 -> C) + BN + ReLU.
  std::vector<float> x = conv2d(input, 4, kSize, kSize, weights_.inputConvW, weights_.inputConvB,
                                C, 3, 3, 1);
  x = batchNorm(x, C, kSize, kSize, weights_.inputBnGamma, weights_.inputBnBeta,
                weights_.inputBnMean, weights_.inputBnVar);
  x = relu(std::move(x));

  // Residual tower.
  for (int b = 0; b < N; ++b) {
    const Weights::Block& block = weights_.blocks[static_cast<std::size_t>(b)];
    std::vector<float> h = conv2d(x, C, kSize, kSize, block.conv1W, block.conv1B, C, 3, 3, 1);
    h = batchNorm(h, C, kSize, kSize, block.bn1Gamma, block.bn1Beta, block.bn1Mean,
                  block.bn1Var);
    h = relu(std::move(h));
    h = conv2d(h, C, kSize, kSize, block.conv2W, block.conv2B, C, 3, 3, 1);
    h = batchNorm(h, C, kSize, kSize, block.bn2Gamma, block.bn2Beta, block.bn2Mean,
                  block.bn2Var);
    for (std::size_t i = 0; i < x.size(); ++i) {
      x[i] = std::max(0.0f, x[i] + h[i]);  // residual add + ReLU
    }
  }

  // Policy head: conv(2) -> flatten(450) -> linear(450 -> 225). conv2d already
  // returns channel-major flattened data, so pass it directly.
  std::vector<float> p =
      conv2d(x, C, kSize, kSize, weights_.policyConvW, weights_.policyConvB, 2, 3, 3, 1);
  const std::vector<float> logits =
      linear(p, weights_.policyFcW, weights_.policyFcB, 225, 2 * HW);

  // Value head: conv(1) -> flatten(225) -> linear(225 -> 256) -> ReLU ->
  // linear(256 -> 1) -> tanh.
  std::vector<float> v =
      conv2d(x, C, kSize, kSize, weights_.valueConvW, weights_.valueConvB, 1, 3, 3, 1);
  std::vector<float> h1 = linear(v, weights_.valueFc1W, weights_.valueFc1B, 256, HW);
  h1 = relu(std::move(h1));
  const std::vector<float> valueOut =
      linear(h1, weights_.valueFc2W, weights_.valueFc2B, 1, 256);

  Output out;
  for (int i = 0; i < 225; ++i) {
    out.policyLogits[static_cast<std::size_t>(i)] = logits[static_cast<std::size_t>(i)];
  }
  out.value = std::tanh(valueOut[0]);
  return out;
}

}  // namespace gomoku
