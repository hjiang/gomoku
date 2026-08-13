#pragma once

#include "core/types.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace gomoku {

// A single training position: the network input planes, the policy target, and
// the value target. See docs/plans/PLAN-stage3-training.md for the
// authoritative binary layout.
struct PositionRecord {
  // Network input, exactly BoardEncoder::encode(board, toMove): 4 planes of
  // kSize*kSize, channel-major, each plane row-major (idx = r*kSize + c).
  std::array<float, 4 * kSize * kSize> planes{};
  // Policy target over cells (row-major, sums to 1). Self-play: normalized
  // root visit counts. Bootstrap: label-smoothed one-hot over the chosen move.
  std::array<float, kSize * kSize> policy{};
  // Game outcome from the perspective of the player to move: +1 win, -1 loss,
  // 0 draw.
  float value = 0.0f;
};

// One recorded game: an ordered list of positions (the board before each move).
// The player to move at position i is Black when i is even and alternates, so
// it is recoverable from the record alone.
struct GameRecord {
  std::vector<PositionRecord> positions;
};

// Binary record stream, little-endian:
//   Stream      := magic[9] "GOMOKUREC"  u32 version (= 1)  Game*
//   Game        := u32 numPositions  PositionRecord[numPositions]
//   PositionRecord := f32 planes[900]  f32 policy[225]  f32 value
// This is the contract shared with the Python trainer; the tests pin it.
[[nodiscard]] std::vector<std::uint8_t> encodeStream(const std::vector<GameRecord>& games);

// Parses a record stream. Returns nullopt on any malformed input: bad magic,
// unsupported version, truncation, a numPositions that cannot fit the buffer,
// or trailing bytes that do not form a game.
[[nodiscard]] std::optional<std::vector<GameRecord>> decodeStream(
    std::span<const std::uint8_t> bytes);

// Label-smoothed one-hot policy target for a supervised (bootstrap) position:
//   target[c] = (1 - eps) * [c == moveIdx] + eps / |legal|   for legal c
//   target[c] = 0                                            otherwise
// Pre: legal[moveIdx] is true, at least one legal cell exists, and eps is in
// [0, 1]. Post: the target sums to 1 and carries no mass on illegal cells.
[[nodiscard]] std::array<float, kSize * kSize> labelSmoothedPolicy(
    int moveIdx, const std::array<bool, kSize * kSize>& legal, float eps = 0.1f);

}  // namespace gomoku
