#include "core/GameRecord.hpp"
#include "core/MctsEngine.hpp"
#include "core/SearchEngine.hpp"
#include "core/SelfPlay.hpp"
#include "core/WinDetector.hpp"

#include "test_gnn_model.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>

using namespace gomoku;

namespace {

// Reconstructs the board shown in position `i` of a record from its planes.
// Records start with Black to move and alternate, so the player to move at
// position i is Black exactly when i is even.
Board boardFromRecord(const GameRecord& rec, std::size_t i) {
  const PositionRecord& pos = rec.positions[i];
  const Player toMove = (i % 2 == 0) ? Player::Black : Player::White;
  Board b;
  for (int r = 0; r < kSize; ++r) {
    for (int c = 0; c < kSize; ++c) {
      const std::size_t idx =
          static_cast<std::size_t>(r) * kSize + static_cast<std::size_t>(c);
      if (pos.planes[idx] == 1.0f) {
        b.place({Position{r, c}, toMove});
      } else if (pos.planes[kSize * kSize + idx] == 1.0f) {
        b.place({Position{r, c}, opponent(toMove)});
      }
    }
  }
  return b;
}

// The single cell added between position i and position i+1 (the move made at
// position i), or nullopt if they do not differ by exactly one cell.
std::optional<Position> moveCellFromRecord(const GameRecord& rec, std::size_t i) {
  const Board before = boardFromRecord(rec, i);
  const Board after = boardFromRecord(rec, i + 1);
  std::optional<Position> move;
  for (int r = 0; r < kSize; ++r) {
    for (int c = 0; c < kSize; ++c) {
      const Position pos{r, c};
      if (before.isEmpty(pos) != after.isEmpty(pos)) {
        if (move.has_value()) {
          return std::nullopt;  // more than one cell changed
        }
        move = pos;
      }
    }
  }
  return move;
}

// The winner of the recorded game, replayed from the planes. The last move is
// the argmax-policy cell (row-major tie-break), which is the actually-played
// move for both label-smoothed bootstrap policies and visit-count self-play
// policies. Returns Player::None for a draw.
Player replayWinner(const GameRecord& rec) {
  const std::size_t last = rec.positions.size() - 1;
  const PositionRecord& pos = rec.positions[last];

  std::size_t best = 0;
  for (std::size_t i = 1; i < pos.policy.size(); ++i) {
    if (pos.policy[i] > pos.policy[best]) {
      best = i;
    }
  }
  const Position lastMove{static_cast<int>(best / kSize), static_cast<int>(best % kSize)};
  const Player lastToMove = (last % 2 == 0) ? Player::Black : Player::White;

  Board finalBoard = boardFromRecord(rec, last);
  REQUIRE(finalBoard.isEmpty(lastMove));
  finalBoard.place({lastMove, lastToMove});

  const Player winner = WinDetector::winnerOf(finalBoard, lastMove);
  REQUIRE((winner != Player::None || finalBoard.isFull()));
  return winner;
}

// Checks every invariant a generated record must satisfy: non-empty, policies
// sum to 1, values in {-1,0,1}, strict alternation with legal moves, and the
// value sign matching the actual game outcome.
void checkGameConsistency(const GameRecord& rec) {
  REQUIRE(!rec.positions.empty());

  for (std::size_t i = 0; i < rec.positions.size(); ++i) {
    const PositionRecord& pos = rec.positions[i];

    float policySum = 0.0f;
    for (float v : pos.policy) {
      policySum += v;
    }
    REQUIRE(policySum == Catch::Approx(1.0f).margin(1e-4f));

    REQUIRE((pos.value == Catch::Approx(1.0f).margin(1e-6f) ||
             pos.value == Catch::Approx(0.0f).margin(1e-6f) ||
             pos.value == Catch::Approx(-1.0f).margin(1e-6f)));

    if (i + 1 < rec.positions.size()) {
      const std::optional<Position> move = moveCellFromRecord(rec, i);
      REQUIRE(move.has_value());
      REQUIRE(boardFromRecord(rec, i).isEmpty(*move));
      REQUIRE(pos.policy[static_cast<std::size_t>(move->row) * kSize + move->col] > 0.0f);
      // The last-move marker of the following position points at the move.
      const std::size_t marker = 3 * kSize * kSize +
                                 static_cast<std::size_t>(move->row) * kSize +
                                 static_cast<std::size_t>(move->col);
      REQUIRE(rec.positions[i + 1].planes[marker] == 1.0f);
    }
  }

  const Player winner = replayWinner(rec);
  for (std::size_t i = 0; i < rec.positions.size(); ++i) {
    const Player toMove = (i % 2 == 0) ? Player::Black : Player::White;
    const float expected = winner == Player::None ? 0.0f : (winner == toMove ? 1.0f : -1.0f);
    REQUIRE(rec.positions[i].value == Catch::Approx(expected).margin(1e-6f));
  }
}

}  // namespace

TEST_CASE("selfplay: bootstrap games are consistent and deterministic per seed", "[selfplay]") {
  SearchParams params;
  // Depth 1 + a budget the search can never hit: the wall-clock deadline must
  // never cut the search mid-depth, or two identical calls could pick different
  // moves (the alpha-beta deadline is the only nondeterminism in the pipeline).
  params.maxDepth = 1;
  params.timeBudgetMs = 5000;
  for (std::uint32_t seed = 1; seed <= 3; ++seed) {
    const GameRecord a = generateBootstrapGame(seed, params);
    const GameRecord b = generateBootstrapGame(seed, params);
    REQUIRE(encodeStream({a}) == encodeStream({b}));
    checkGameConsistency(a);
  }
}

TEST_CASE("selfplay: self-play games are consistent and deterministic per seed", "[selfplay]") {
  MctsEngine::unloadModel();
  REQUIRE(MctsEngine::loadModel(makeZeroGnn(1, 2)));

  SearchParams params;
  params.mctsSimulations = 30;
  params.timeBudgetMs = 10000;  // sims must be the binding constraint

  for (std::uint32_t seed = 1; seed <= 2; ++seed) {
    const GameRecord a = generateSelfPlayGame(seed, params);
    const GameRecord b = generateSelfPlayGame(seed, params);
    REQUIRE(encodeStream({a}) == encodeStream({b}));
    checkGameConsistency(a);
  }
  MctsEngine::unloadModel();
}

TEST_CASE("selfplay: self-play without a loaded model throws", "[selfplay]") {
  MctsEngine::unloadModel();
  SearchParams params;
  params.mctsSimulations = 10;
  REQUIRE_THROWS_AS(generateSelfPlayGame(1, params), std::runtime_error);
}

TEST_CASE("selfplay: generated games survive the byte-stream round-trip", "[selfplay]") {
  SearchParams params;
  params.maxDepth = 2;
  params.timeBudgetMs = 20;
  const GameRecord game = generateBootstrapGame(7, params);

  const std::vector<std::uint8_t> bytes = encodeStream({game});
  const auto decoded = decodeStream(bytes);
  REQUIRE(decoded.has_value());
  REQUIRE(decoded->size() == 1);
  REQUIRE((*decoded)[0].positions.size() == game.positions.size());
  for (std::size_t i = 0; i < game.positions.size(); ++i) {
    REQUIRE((*decoded)[0].positions[i].planes == game.positions[i].planes);
    REQUIRE((*decoded)[0].positions[i].policy == game.positions[i].policy);
    REQUIRE((*decoded)[0].positions[i].value == Catch::Approx(game.positions[i].value));
  }
}
