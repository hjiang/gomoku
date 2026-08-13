#include "core/GameRecord.hpp"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstdint>
#include <cstring>

namespace gomoku {
namespace {

constexpr std::array<std::uint8_t, 9> kMagic = {'G', 'O', 'M', 'O', 'K', 'U', 'R', 'E', 'C'};
constexpr std::uint32_t kFormatVersion = 1;

// Floats per position: 900 planes + 225 policy + 1 value = 1126.
constexpr std::size_t kPosFloats = 4 * kSize * kSize + kSize * kSize + 1;
constexpr std::size_t kPosBytes = kPosFloats * sizeof(float);

void appendBytes(std::vector<std::uint8_t>& out, const void* data, std::size_t n) {
  const auto* p = static_cast<const std::uint8_t*>(data);
  out.insert(out.end(), p, p + n);
}

}  // namespace

std::vector<std::uint8_t> encodeStream(const std::vector<GameRecord>& games) {
  static_assert(std::endian::native == std::endian::little,
                "record stream stores little-endian f32");

  std::size_t total = kMagic.size() + sizeof(std::uint32_t);
  for (const GameRecord& game : games) {
    total += sizeof(std::uint32_t) + game.positions.size() * kPosBytes;
  }
  std::vector<std::uint8_t> out;
  out.reserve(total);

  appendBytes(out, kMagic.data(), kMagic.size());
  const std::uint32_t version = kFormatVersion;
  appendBytes(out, &version, sizeof(version));

  for (const GameRecord& game : games) {
    const std::uint32_t numPositions = static_cast<std::uint32_t>(game.positions.size());
    appendBytes(out, &numPositions, sizeof(numPositions));
    for (const PositionRecord& pos : game.positions) {
      appendBytes(out, pos.planes.data(), sizeof(pos.planes));
      appendBytes(out, pos.policy.data(), sizeof(pos.policy));
      appendBytes(out, &pos.value, sizeof(pos.value));
    }
  }
  return out;
}

std::optional<std::vector<GameRecord>> decodeStream(std::span<const std::uint8_t> bytes) {
  static_assert(std::endian::native == std::endian::little,
                "record stream stores little-endian f32");

  if (bytes.size() < kMagic.size() ||
      !std::equal(kMagic.begin(), kMagic.end(), bytes.begin())) {
    return std::nullopt;
  }
  std::size_t off = kMagic.size();

  auto readU32 = [&](std::uint32_t& out) {
    if (off + sizeof(out) > bytes.size()) {
      return false;
    }
    std::memcpy(&out, bytes.data() + off, sizeof(out));
    off += sizeof(out);
    return true;
  };
  auto readF32s = [&](std::span<float> dst) {
    if (off + dst.size() * sizeof(float) > bytes.size()) {
      return false;
    }
    std::memcpy(dst.data(), bytes.data() + off, dst.size() * sizeof(float));
    off += dst.size() * sizeof(float);
    return true;
  };

  std::uint32_t version = 0;
  if (!readU32(version) || version != kFormatVersion) {
    return std::nullopt;
  }

  std::vector<GameRecord> games;
  while (off < bytes.size()) {
    std::uint32_t numPositions = 0;
    if (!readU32(numPositions)) {
      return std::nullopt;
    }
    // A numPositions that cannot fit the remaining bytes is malformed; this
    // also bounds the allocation below.
    if (static_cast<std::size_t>(numPositions) > (bytes.size() - off) / kPosBytes) {
      return std::nullopt;
    }
    GameRecord game;
    game.positions.resize(numPositions);
    for (PositionRecord& pos : game.positions) {
      if (!readF32s(pos.planes) || !readF32s(pos.policy) ||
          !readF32s(std::span<float>(&pos.value, 1))) {
        return std::nullopt;
      }
    }
    games.push_back(std::move(game));
  }
  return games;
}

std::array<float, kSize * kSize> labelSmoothedPolicy(
    int moveIdx, const std::array<bool, kSize * kSize>& legal, float eps) {
  assert(moveIdx >= 0 && moveIdx < kSize * kSize);
  assert(eps >= 0.0f && eps <= 1.0f);
  assert(legal[static_cast<std::size_t>(moveIdx)]);

  int numLegal = 0;
  for (const bool l : legal) {
    numLegal += l ? 1 : 0;
  }
  assert(numLegal > 0);

  std::array<float, kSize * kSize> target{};
  const float legalShare = eps / static_cast<float>(numLegal);
  for (std::size_t i = 0; i < legal.size(); ++i) {
    if (legal[i]) {
      target[i] = legalShare;
    }
  }
  target[static_cast<std::size_t>(moveIdx)] += 1.0f - eps;
  return target;
}

}  // namespace gomoku
