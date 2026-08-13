#include "core/GameRecord.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstdint>
#include <vector>

using namespace gomoku;

namespace {

// A position record with recognizable, exactly-representable values.
PositionRecord makePosition(float planeSeed) {
  PositionRecord rec;
  for (std::size_t i = 0; i < rec.planes.size(); ++i) {
    rec.planes[i] = static_cast<float>(static_cast<int>(i % 11)) + planeSeed;
  }
  for (std::size_t i = 0; i < rec.policy.size(); ++i) {
    rec.policy[i] = static_cast<float>(static_cast<int>(i % 5));
  }
  rec.value = 1.0f;
  return rec;
}

}  // namespace

TEST_CASE("game record: empty stream round-trips", "[record]") {
  const std::vector<std::uint8_t> bytes = encodeStream({});
  const auto decoded = decodeStream(bytes);
  REQUIRE(decoded.has_value());
  REQUIRE(decoded->empty());
}

TEST_CASE("game record: encode/decode round-trips games and positions", "[record]") {
  std::vector<GameRecord> games(3);
  games[0].positions = {makePosition(0.0f), makePosition(1.0f)};
  games[1].positions.clear();  // a zero-position game is still well-formed
  games[2].positions = {makePosition(2.5f)};

  const std::vector<std::uint8_t> bytes = encodeStream(games);
  const auto decoded = decodeStream(bytes);
  REQUIRE(decoded.has_value());
  REQUIRE(decoded->size() == games.size());

  for (std::size_t g = 0; g < games.size(); ++g) {
    REQUIRE((*decoded)[g].positions.size() == games[g].positions.size());
    for (std::size_t p = 0; p < games[g].positions.size(); ++p) {
      const PositionRecord& a = games[g].positions[p];
      const PositionRecord& b = (*decoded)[g].positions[p];
      REQUIRE(a.planes == b.planes);
      REQUIRE(a.policy == b.policy);
      REQUIRE(a.value == Catch::Approx(b.value));
    }
  }
}

TEST_CASE("game record: rejects a bad magic", "[record]") {
  std::vector<std::uint8_t> bytes = encodeStream({});
  bytes[0] = 'X';
  REQUIRE_FALSE(decodeStream(bytes).has_value());
}

TEST_CASE("game record: rejects an unsupported version", "[record]") {
  std::vector<std::uint8_t> bytes = encodeStream({});
  bytes[9] = 99;  // first byte of the little-endian version field
  REQUIRE_FALSE(decodeStream(bytes).has_value());
}

TEST_CASE("game record: rejects truncation", "[record]") {
  std::vector<GameRecord> games;
  games.push_back(GameRecord{{makePosition(0.0f)}});
  std::vector<std::uint8_t> bytes = encodeStream(games);
  bytes.resize(bytes.size() - 4);  // chop part of a position
  REQUIRE_FALSE(decodeStream(bytes).has_value());
}

TEST_CASE("game record: rejects a numPositions that exceeds the buffer", "[record]") {
  std::vector<GameRecord> games;
  games.push_back(GameRecord{{makePosition(0.0f)}});
  std::vector<std::uint8_t> bytes = encodeStream(games);
  // numPositions is the u32 right after the 13-byte header (9 magic + 4 version).
  for (int i = 0; i < 4; ++i) {
    bytes[13 + static_cast<std::size_t>(i)] = 0xFF;
  }
  REQUIRE_FALSE(decodeStream(bytes).has_value());
}

TEST_CASE("game record: rejects trailing garbage", "[record]") {
  std::vector<std::uint8_t> bytes = encodeStream({});
  bytes.push_back(0x00);
  REQUIRE_FALSE(decodeStream(bytes).has_value());
}

TEST_CASE("game record: label smoothing concentrates on the chosen move", "[record]") {
  std::array<bool, kSize * kSize> legal{};
  legal.fill(false);
  legal[0] = true;
  legal[100] = true;
  legal[224] = true;

  const auto p = labelSmoothedPolicy(100, legal);
  REQUIRE(p[100] == Catch::Approx(1.0f - 0.1f + 0.1f / 3.0f).margin(1e-6f));
  REQUIRE(p[0] == Catch::Approx(0.1f / 3.0f).margin(1e-6f));
  REQUIRE(p[224] == Catch::Approx(0.1f / 3.0f).margin(1e-6f));
  REQUIRE(p[1] == 0.0f);    // illegal cell: no mass
  REQUIRE(p[223] == 0.0f);  // illegal cell: no mass

  float sum = 0.0f;
  for (float v : p) {
    sum += v;
  }
  REQUIRE(sum == Catch::Approx(1.0f).margin(1e-5f));
}

TEST_CASE("game record: label smoothing over a single legal cell is one-hot", "[record]") {
  std::array<bool, kSize * kSize> legal{};
  legal.fill(false);
  legal[50] = true;

  const auto p = labelSmoothedPolicy(50, legal);
  REQUIRE(p[50] == Catch::Approx(1.0f).margin(1e-6f));
  float sum = 0.0f;
  for (float v : p) {
    sum += v;
  }
  REQUIRE(sum == Catch::Approx(1.0f).margin(1e-6f));
}
