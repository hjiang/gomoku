#include "core/Weights.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>

namespace gomoku {

namespace {

constexpr std::size_t kMagicLen = 9;
constexpr std::uint32_t kFormatVersion = 1;

// Fixed float counts that do not depend on numBlocks/channels.
constexpr std::size_t kFixedFloats =
    2 /* policyConvB */ + 1 /* valueConvB */ + 225 * 450 /* policyFcW */ + 225 /* policyFcB */ +
    256 * 225 /* valueFc1W */ + 256 /* valueFc1B */ + 256 /* valueFc2W */ + 1 /* valueFc2B */;

// Total float count for numBlocks N and channels C (see the tensor order in
// `parse`). = N*(18*C^2 + 10*C) + 68*C + kFixedFloats.
std::size_t totalFloats(std::size_t numBlocks, std::size_t channels) {
  const std::size_t C2 = channels * channels;
  const std::size_t perBlock = 2 * C2 * 9 + 10 * channels;
  const std::size_t perChannel = 36 * channels  /* inputConvW */
                                 + channels     /* inputConvB */
                                 + 4 * channels /* input BN */
                                 + 18 * channels /* policyConvW */
                                 + 9 * channels /* valueConvW */;
  return numBlocks * perBlock + perChannel + kFixedFloats;
}

}  // namespace

std::optional<Weights> Weights::parse(std::span<const std::uint8_t> bytes) {
  static_assert(std::endian::native == std::endian::little,
                ".gnn loader reads little-endian tensors");

  constexpr std::array<std::uint8_t, kMagicLen> kMagic = {'G', 'O', 'M', 'O', 'K',
                                                          'U', 'N', 'E', 'T'};
  constexpr std::size_t kHeaderLen = kMagicLen + 3 * sizeof(std::uint32_t);
  if (bytes.size() < kHeaderLen) {
    return std::nullopt;
  }
  if (!std::equal(kMagic.begin(), kMagic.end(), bytes.begin())) {
    return std::nullopt;
  }

  std::size_t off = kMagicLen;
  std::uint32_t version = 0, numBlocks = 0, channels = 0;
  std::memcpy(&version, bytes.data() + off, 4);
  off += 4;
  std::memcpy(&numBlocks, bytes.data() + off, 4);
  off += 4;
  std::memcpy(&channels, bytes.data() + off, 4);
  off += 4;

  if (version != kFormatVersion) {
    return std::nullopt;
  }
  if (numBlocks < 1 || channels < 1 || numBlocks > 1024 || channels > 1024) {
    return std::nullopt;
  }

  const std::size_t N = numBlocks;
  const std::size_t C = channels;
  if (bytes.size() != kHeaderLen + totalFloats(N, C) * sizeof(float)) {
    return std::nullopt;
  }

  auto readF32s = [&](std::size_t count) {
    std::vector<float> v(count);
    std::memcpy(v.data(), bytes.data() + off, count * sizeof(float));
    off += count * sizeof(float);
    return v;
  };

  Weights w;
  w.numBlocks = numBlocks;
  w.channels = channels;

  w.inputConvW = readF32s(C * 4 * 9);
  w.inputConvB = readF32s(C);
  w.inputBnGamma = readF32s(C);
  w.inputBnBeta = readF32s(C);
  w.inputBnMean = readF32s(C);
  w.inputBnVar = readF32s(C);

  w.blocks.reserve(N);
  for (std::size_t b = 0; b < N; ++b) {
    Weights::Block block;
    block.conv1W = readF32s(C * C * 9);
    block.conv1B = readF32s(C);
    block.bn1Gamma = readF32s(C);
    block.bn1Beta = readF32s(C);
    block.bn1Mean = readF32s(C);
    block.bn1Var = readF32s(C);
    block.conv2W = readF32s(C * C * 9);
    block.conv2B = readF32s(C);
    block.bn2Gamma = readF32s(C);
    block.bn2Beta = readF32s(C);
    block.bn2Mean = readF32s(C);
    block.bn2Var = readF32s(C);
    w.blocks.push_back(std::move(block));
  }

  w.policyConvW = readF32s(2 * C * 9);
  w.policyConvB = readF32s(2);
  w.policyFcW = readF32s(225 * 450);
  w.policyFcB = readF32s(225);

  w.valueConvW = readF32s(C * 9);
  w.valueConvB = readF32s(1);
  w.valueFc1W = readF32s(256 * 225);
  w.valueFc1B = readF32s(256);
  w.valueFc2W = readF32s(256);
  w.valueFc2B = readF32s(1);

  return w;
}

}  // namespace gomoku
