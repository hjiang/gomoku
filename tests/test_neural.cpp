#include "core/Board.hpp"
#include "core/BoardEncoder.hpp"
#include "core/NeuralNet.hpp"
#include "core/TensorOps.hpp"
#include "core/Weights.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

using namespace gomoku;

namespace {

// ---------------------------------------------------------------------------
// .gnn byte-buffer builder for tests. Appends the header and every tensor in
// the exact order the Weights::parse loader reads, all zero-initialized.
// ---------------------------------------------------------------------------
void appendU32(std::vector<std::uint8_t>& buf, std::uint32_t v) {
  for (int i = 0; i < 4; ++i) {
    buf.push_back(static_cast<std::uint8_t>((v >> (8 * i)) & 0xFF));
  }
}

void appendF32(std::vector<std::uint8_t>& buf, float v) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &v, sizeof(bits));
  appendU32(buf, bits);
}

// Offset of the input-conv bias tensor: 21-byte header (9 magic + 3 u32) then
// the input conv weight tensor of C*4*3*3 floats.
size_t inputConvBiasOffset(int channels) {
  return 21 + static_cast<size_t>(channels) * 4 * 3 * 3 * 4;
}

// Byte offset of the policy fc bias tensor (225 floats) for numBlocks/channels,
// mirroring the exact tensor order of makeGnnBytes / Weights::parse.
size_t policyFcBOffset(int numBlocks, int channels) {
  const size_t C = static_cast<size_t>(channels);
  const size_t N = static_cast<size_t>(numBlocks);
  const size_t preceding =
      36 * C + C + 4 * C +          // input conv W, bias, input BN
      N * (18 * C * C + 10 * C) +   // residual blocks
      18 * C + 2 + 225 * 450;       // policy conv W, bias, policy fc W
  return 21 + preceding * 4;
}

// Byte offset of the value fc2 bias tensor (1 float).
size_t valueFc2BOffset(int numBlocks, int channels) {
  const size_t C = static_cast<size_t>(channels);
  const size_t N = static_cast<size_t>(numBlocks);
  const size_t preceding =
      36 * C + C + 4 * C +          // input conv W, bias, input BN
      N * (18 * C * C + 10 * C) +   // residual blocks
      18 * C + 2 + 225 * 450 +      // policy conv W, bias, policy fc W
      225 +                         // policy fc bias
      9 * C + 1 +                   // value conv W, bias
      256 * 225 + 256 + 256;        // value fc1 W, bias, value fc2 W
  return 21 + preceding * 4;
}

void patchF32(std::vector<std::uint8_t>& buf, size_t offset, float v) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &v, sizeof(bits));
  for (int i = 0; i < 4; ++i) {
    buf[offset + i] = static_cast<std::uint8_t>((bits >> (8 * i)) & 0xFF);
  }
}

// Builds a full .gnn buffer for `numBlocks` blocks of `channels` channels.
std::vector<std::uint8_t> makeGnnBytes(int numBlocks, int channels) {
  std::vector<std::uint8_t> buf;
  const char* magic = "GOMOKUNET";
  buf.insert(buf.end(), magic, magic + 9);
  appendU32(buf, 1);  // version
  appendU32(buf, static_cast<std::uint32_t>(numBlocks));
  appendU32(buf, static_cast<std::uint32_t>(channels));

  const int C = channels;
  const auto zeros = [&](size_t count) {
    for (size_t i = 0; i < count; ++i) {
      appendF32(buf, 0.0f);
    }
  };

  zeros(C * 4 * 9);            // input conv weight (C,4,3,3)
  zeros(C);                    // input conv bias
  zeros(C); zeros(C); zeros(C); zeros(C);  // input bn gamma/beta/mean/var
  for (int b = 0; b < numBlocks; ++b) {
    zeros(C * C * 9); zeros(C);                       // conv1 W, b
    zeros(C); zeros(C); zeros(C); zeros(C);           // bn1 gamma/beta/mean/var
    zeros(C * C * 9); zeros(C);                       // conv2 W, b
    zeros(C); zeros(C); zeros(C); zeros(C);           // bn2 gamma/beta/mean/var
  }
  zeros(2 * C * 9);            // policy conv weight (2,C,3,3)
  zeros(2);                    // policy conv bias
  zeros(225 * 450);            // policy fc weight
  zeros(225);                  // policy fc bias
  zeros(C * 9);                // value conv weight (1,C,3,3)
  zeros(1);                    // value conv bias
  zeros(256 * 225);            // value fc1 weight
  zeros(256);                  // value fc1 bias
  zeros(256);                  // value fc2 weight (1,256)
  zeros(1);                    // value fc2 bias
  return buf;
}

}  // namespace

// ---------------------------------------------------------------------------
// BoardEncoder
// ---------------------------------------------------------------------------
TEST_CASE("encoder: planes are correct for Black to move", "[neural][encoder]") {
  Board b;
  b.place({Position{3, 4}, Player::Black});
  b.place({Position{7, 7}, Player::White});
  b.place({Position{0, 0}, Player::Black});  // last move

  const auto enc = BoardEncoder::encode(b, Player::Black);
  REQUIRE(enc.size() == BoardEncoder::kInputSize);
  const int HW = kSize * kSize;

  // Plane 0: Black stones.
  REQUIRE(enc[3 * kSize + 4] == 1.0f);
  REQUIRE(enc[0 * kSize + 0] == 1.0f);
  REQUIRE(enc[7 * kSize + 7] == 0.0f);  // White is not on plane 0
  // Plane 1: White stones.
  REQUIRE(enc[HW + 7 * kSize + 7] == 1.0f);
  REQUIRE(enc[HW + 3 * kSize + 4] == 0.0f);
  // Plane 2: constant fill.
  for (int i = 0; i < HW; ++i) {
    REQUIRE(enc[2 * HW + i] == 1.0f);
  }
  // Plane 3: last-move marker only at (0,0).
  REQUIRE(enc[3 * HW + 0] == 1.0f);
  REQUIRE(enc[3 * HW + 7 * kSize + 7] == 0.0f);
}

TEST_CASE("encoder: planes flip when encoding for White", "[neural][encoder]") {
  Board b;
  b.place({Position{3, 4}, Player::Black});
  b.place({Position{7, 7}, Player::White});  // last move

  const auto enc = BoardEncoder::encode(b, Player::White);
  const int HW = kSize * kSize;

  // Plane 0 now holds White stones.
  REQUIRE(enc[7 * kSize + 7] == 1.0f);
  REQUIRE(enc[3 * kSize + 4] == 0.0f);
  // Plane 1 now holds Black stones.
  REQUIRE(enc[HW + 3 * kSize + 4] == 1.0f);
  REQUIRE(enc[HW + 7 * kSize + 7] == 0.0f);
  // Last-move marker is absolute, still at (7,7).
  REQUIRE(enc[3 * HW + 7 * kSize + 7] == 1.0f);
}

TEST_CASE("encoder: last-move plane is all zero on an empty board", "[neural][encoder]") {
  Board b;
  const auto enc = BoardEncoder::encode(b, Player::Black);
  const int HW = kSize * kSize;
  for (int i = 0; i < HW; ++i) {
    REQUIRE(enc[3 * HW + i] == 0.0f);
  }
}

// ---------------------------------------------------------------------------
// Weights::parse
// ---------------------------------------------------------------------------
TEST_CASE("weights: minimal C=1 N=1 buffer round-trips", "[neural][weights]") {
  std::vector<std::uint8_t> buf = makeGnnBytes(1, 1);
  patchF32(buf, inputConvBiasOffset(1), 3.5f);

  const auto w = Weights::parse(buf);
  REQUIRE(w.has_value());
  REQUIRE(w->numBlocks == 1);
  REQUIRE(w->channels == 1);
  REQUIRE(w->inputConvB.size() == 1);
  REQUIRE(w->inputConvB[0] == Catch::Approx(3.5f));
}

TEST_CASE("weights: larger layout keeps tensor sizes", "[neural][weights]") {
  std::vector<std::uint8_t> buf = makeGnnBytes(2, 3);
  const auto w = Weights::parse(buf);
  REQUIRE(w.has_value());
  REQUIRE(w->numBlocks == 2);
  REQUIRE(w->channels == 3);
  REQUIRE(w->inputConvW.size() == 3 * 4 * 9);
  REQUIRE(w->blocks.size() == 2);
  REQUIRE(w->blocks[0].conv1W.size() == 3 * 3 * 9);
  REQUIRE(w->policyFcW.size() == 225 * 450);
  REQUIRE(w->valueFc1W.size() == 256 * 225);
}

TEST_CASE("weights: rejects malformed input", "[neural][weights]") {
  // Bad magic.
  std::vector<std::uint8_t> badMagic = makeGnnBytes(1, 1);
  badMagic[0] = 'X';
  REQUIRE_FALSE(Weights::parse(badMagic).has_value());

  // Wrong version.
  std::vector<std::uint8_t> badVersion = makeGnnBytes(1, 1);
  patchF32(badVersion, 9, 99.0f);
  REQUIRE_FALSE(Weights::parse(badVersion).has_value());

  // Truncated buffer.
  std::vector<std::uint8_t> truncated = makeGnnBytes(2, 2);
  truncated.resize(truncated.size() / 2);
  REQUIRE_FALSE(Weights::parse(truncated).has_value());

  // Zero blocks.
  std::vector<std::uint8_t> noBlocks = makeGnnBytes(0, 1);
  REQUIRE_FALSE(Weights::parse(noBlocks).has_value());

  // Too small to even hold the header.
  REQUIRE_FALSE(Weights::parse(std::vector<std::uint8_t>(5, 0)).has_value());
}

// ---------------------------------------------------------------------------
// TensorOps
// ---------------------------------------------------------------------------
TEST_CASE("tensor ops: conv2d impulse response", "[neural][tensorops]") {
  // 1 channel, 3x3 input with a single impulse at the center; 3x3 kernel,
  // pad 1 => 3x3 output. out[y][x] = W[2-y][2-x] (kernel rotated 180 deg).
  std::vector<float> input(9, 0.0f);
  input[1 * 3 + 1] = 1.0f;
  const std::vector<float> weight = {1, 2, 3, 4, 5, 6, 7, 8, 9};  // [dy][dx]
  const std::vector<float> bias = {0.0f};

  const auto out = conv2d(input, 1, 3, 3, weight, bias, 1, 3, 3, 1);
  REQUIRE(out.size() == 9);
  REQUIRE(out[0] == Catch::Approx(9.0f));  // W[2][2]
  REQUIRE(out[1] == Catch::Approx(8.0f));  // W[2][1]
  REQUIRE(out[2] == Catch::Approx(7.0f));  // W[2][0]
  REQUIRE(out[3] == Catch::Approx(6.0f));  // W[1][2]
  REQUIRE(out[4] == Catch::Approx(5.0f));  // W[1][1]
  REQUIRE(out[5] == Catch::Approx(4.0f));  // W[1][0]
  REQUIRE(out[6] == Catch::Approx(3.0f));  // W[0][2]
  REQUIRE(out[7] == Catch::Approx(2.0f));  // W[0][1]
  REQUIRE(out[8] == Catch::Approx(1.0f));  // W[0][0]
}

TEST_CASE("tensor ops: conv2d applies bias", "[neural][tensorops]") {
  std::vector<float> input(9, 0.0f);
  input[1 * 3 + 1] = 1.0f;
  const std::vector<float> weight = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  const std::vector<float> bias = {1.5f};

  const auto out = conv2d(input, 1, 3, 3, weight, bias, 1, 3, 3, 1);
  REQUIRE(out[4] == Catch::Approx(5.0f + 1.5f));
}

TEST_CASE("tensor ops: linear with identity weight is in + bias", "[neural][tensorops]") {
  const std::vector<float> input = {5.0f, 6.0f, 7.0f};
  const std::vector<float> weight = {1, 0, 0, 0, 1, 0, 0, 0, 1};  // 3x3 identity
  const std::vector<float> bias = {1.0f, 2.0f, 3.0f};

  const auto out = linear(input, weight, bias, 3, 3);
  REQUIRE(out.size() == 3);
  REQUIRE(out[0] == Catch::Approx(6.0f));
  REQUIRE(out[1] == Catch::Approx(8.0f));
  REQUIRE(out[2] == Catch::Approx(10.0f));
}

TEST_CASE("tensor ops: batchNorm with unit params is identity", "[neural][tensorops]") {
  const std::vector<float> input = {1.0f, -2.0f, 3.5f};
  const std::vector<float> gamma = {1.0f};
  const std::vector<float> beta = {0.0f};
  const std::vector<float> mean = {0.0f};
  const std::vector<float> var = {1.0f};

  const auto out = batchNorm(input, 1, 1, 3, gamma, beta, mean, var);
  REQUIRE(out.size() == 3);
  for (size_t i = 0; i < out.size(); ++i) {
    REQUIRE(out[i] == Catch::Approx(input[i]).margin(1e-4f));
  }
}

TEST_CASE("tensor ops: relu clamps negatives", "[neural][tensorops]") {
  std::vector<float> input = {-3.0f, 0.0f, 2.5f};
  const auto out = relu(std::move(input));
  REQUIRE(out[0] == 0.0f);
  REQUIRE(out[1] == 0.0f);
  REQUIRE(out[2] == Catch::Approx(2.5f));
}

TEST_CASE("tensor ops: softmax sums to one and is monotone", "[neural][tensorops]") {
  const std::vector<float> logits = {1.0f, 2.0f, 3.0f};
  const auto p = softmax(logits);
  REQUIRE(p.size() == 3);
  float sum = 0.0f;
  for (float v : p) {
    sum += v;
  }
  REQUIRE(sum == Catch::Approx(1.0f).margin(1e-5f));
  REQUIRE(p[0] < p[1]);
  REQUIRE(p[1] < p[2]);
}

TEST_CASE("tensor ops: softmax with -inf masks that entry", "[neural][tensorops]") {
  const std::vector<float> logits = {1.0f, -std::numeric_limits<float>::infinity(), 2.0f};
  const auto p = softmax(logits);
  REQUIRE(p.size() == 3);
  REQUIRE(p[1] == 0.0f);
  REQUIRE(p[0] + p[2] == Catch::Approx(1.0f).margin(1e-5f));
  REQUIRE(p[0] < p[2]);
}

// ---------------------------------------------------------------------------
// NeuralNet
// ---------------------------------------------------------------------------
TEST_CASE("neural net: zero network yields zero logits and zero value", "[neural][net]") {
  const auto w = Weights::parse(makeGnnBytes(2, 1));
  REQUIRE(w.has_value());
  const NeuralNet net(*w);

  std::array<float, BoardEncoder::kInputSize> input{};
  input.fill(1.0f);  // nonzero input
  const auto out = net.evaluate(input);

  for (float logit : out.policyLogits) {
    REQUIRE(logit == 0.0f);
  }
  REQUIRE(out.value == Catch::Approx(0.0f));
}

TEST_CASE("neural net: output has 225 logits and a finite value", "[neural][net]") {
  const auto w = Weights::parse(makeGnnBytes(2, 1));
  REQUIRE(w.has_value());
  const NeuralNet net(*w);

  Board b;
  b.place({Position{7, 7}, Player::Black});
  const auto enc = BoardEncoder::encode(b, Player::Black);
  const auto out = net.evaluate(enc);

  REQUIRE(out.policyLogits.size() == kSize * kSize);
  REQUIRE(std::isfinite(out.value));
}

TEST_CASE("neural net: nonzero biases propagate through the composed forward", "[neural][net]") {
  std::vector<std::uint8_t> buf = makeGnnBytes(1, 2);  // N=1, C=2, all zero
  for (int i = 0; i < 225; ++i) {
    patchF32(buf, policyFcBOffset(1, 2) + static_cast<std::size_t>(i) * 4, 0.1f * i);
  }
  patchF32(buf, valueFc2BOffset(1, 2), 0.5f);

  const auto w = Weights::parse(buf);
  REQUIRE(w.has_value());
  const NeuralNet net(*w);

  const auto enc = BoardEncoder::encode(Board{}, Player::Black);
  const auto out = net.evaluate(enc);

  // With all conv/fc weights zero, every activation stays zero, so logits ==
  // policyFcB and value == tanh(valueFc2B). This pins the full composed
  // forward pass with nonzero (bias) weights.
  for (int i = 0; i < 225; ++i) {
    REQUIRE(out.policyLogits[static_cast<std::size_t>(i)] ==
            Catch::Approx(0.1f * i).margin(1e-4f));
  }
  REQUIRE(out.value == Catch::Approx(std::tanh(0.5f)).margin(1e-5f));
}
