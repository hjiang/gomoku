#include "core/Board.hpp"
#include "core/MctsEngine.hpp"
#include "core/WinDetector.hpp"

#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

using namespace gomoku;

namespace {

// Builds a valid .gnn byte buffer with all-zero tensors for `numBlocks`
// blocks of `channels` channels. A zero network gives a uniform policy and
// value 0, so MCTS relies purely on terminal win/loss detection.
std::vector<std::uint8_t> makeZeroGnn(int numBlocks, int channels) {
  std::vector<std::uint8_t> buf;
  const char* magic = "GOMOKUNET";
  buf.insert(buf.end(), magic, magic + 9);

  const auto putU32 = [&](std::uint32_t v) {
    for (int i = 0; i < 4; ++i) {
      buf.push_back(static_cast<std::uint8_t>((v >> (8 * i)) & 0xFF));
    }
  };
  const auto putF32 = [&](float v) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &v, sizeof(bits));
    putU32(bits);
  };
  const auto zeros = [&](std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) {
      putF32(0.0f);
    }
  };

  putU32(1);  // version
  putU32(static_cast<std::uint32_t>(numBlocks));
  putU32(static_cast<std::uint32_t>(channels));

  const int C = channels;
  zeros(C * 4 * 9);                        // input conv weight (C,4,3,3)
  zeros(C);                                // input conv bias
  zeros(C); zeros(C); zeros(C); zeros(C);  // input bn gamma/beta/mean/var
  for (int b = 0; b < numBlocks; ++b) {
    zeros(C * C * 9); zeros(C);                 // conv1 W, b
    zeros(C); zeros(C); zeros(C); zeros(C);     // bn1 gamma/beta/mean/var
    zeros(C * C * 9); zeros(C);                 // conv2 W, b
    zeros(C); zeros(C); zeros(C); zeros(C);     // bn2 gamma/beta/mean/var
  }
  zeros(2 * C * 9);  // policy conv weight (2,C,3,3)
  zeros(2);          // policy conv bias
  zeros(225 * 450);  // policy fc weight
  zeros(225);        // policy fc bias
  zeros(C * 9);      // value conv weight (1,C,3,3)
  zeros(1);          // value conv bias
  zeros(256 * 225);  // value fc1 weight
  zeros(256);        // value fc1 bias
  zeros(256);        // value fc2 weight (1,256)
  zeros(1);          // value fc2 bias
  return buf;
}

// Black has four at (7, 3..6); Black to move wins by playing (7,2), (7,7) or
// (7,8).
Board winInOneFor(Player p) {
  Board b;
  for (int c = 3; c <= 6; ++c) {
    b.place({Position{7, c}, p});
  }
  b.place({Position{1, 1}, opponent(p)});
  b.place({Position{13, 13}, opponent(p)});
  return b;
}

// White has four at (7, 3..6) with the right end capped by Black at (7,7);
// the only move that stops White from winning at (7,2) is Black's (7,2).
Board openFourThreat() {
  Board b;
  for (int c = 3; c <= 6; ++c) {
    b.place({Position{7, c}, Player::White});
  }
  b.place({Position{7, 7}, Player::Black});  // caps the right end
  b.place({Position{5, 5}, Player::Black});
  b.place({Position{9, 9}, Player::Black});
  return b;
}

}  // namespace

TEST_CASE("MCTS: no model loaded throws", "[mcts]") {
  MctsEngine::unloadModel();
  REQUIRE_FALSE(MctsEngine::isModelAvailable());
  SearchParams params;
  params.mctsSimulations = 10;
  REQUIRE_THROWS_AS(MctsEngine::findBestMove(Board{}, Player::Black, params),
                    std::runtime_error);
}

TEST_CASE("MCTS: loadModel with valid bytes enables the engine", "[mcts]") {
  MctsEngine::unloadModel();
  REQUIRE(MctsEngine::loadModel(makeZeroGnn(1, 2)));
  REQUIRE(MctsEngine::isModelAvailable());

  Board b;
  b.place({Position{7, 7}, Player::Black});
  SearchParams params;
  params.mctsSimulations = 50;
  const Move m = MctsEngine::findBestMove(b, Player::White, params);
  REQUIRE(b.inBounds(m.pos));
  REQUIRE(b.isEmpty(m.pos));
  MctsEngine::unloadModel();
}

TEST_CASE("MCTS: loadModel rejects malformed bytes", "[mcts]") {
  MctsEngine::unloadModel();
  REQUIRE_FALSE(MctsEngine::loadModel(std::vector<std::uint8_t>(20, 0)));
  REQUIRE_FALSE(MctsEngine::isModelAvailable());
}

TEST_CASE("MCTS: failed loadModel keeps a previously loaded model", "[mcts]") {
  MctsEngine::unloadModel();
  REQUIRE(MctsEngine::loadModel(makeZeroGnn(1, 2)));
  REQUIRE_FALSE(MctsEngine::loadModel(std::vector<std::uint8_t>(10, 0)));
  REQUIRE(MctsEngine::isModelAvailable());
  MctsEngine::unloadModel();
}

TEST_CASE("MCTS: finds an immediate win", "[mcts]") {
  MctsEngine::unloadModel();
  REQUIRE(MctsEngine::loadModel(makeZeroGnn(1, 2)));
  Board b = winInOneFor(Player::Black);
  SearchParams params;
  params.mctsSimulations = 800;
  params.timeBudgetMs = 5000;
  const Move m = MctsEngine::findBestMove(b, Player::Black, params);
  REQUIRE(b.isEmpty(m.pos));
  b.place(m);
  REQUIRE(WinDetector::winnerOf(b, m.pos) == Player::Black);
  MctsEngine::unloadModel();
}

TEST_CASE("MCTS: blocks an opponent open four", "[mcts]") {
  MctsEngine::unloadModel();
  REQUIRE(MctsEngine::loadModel(makeZeroGnn(1, 2)));
  Board b = openFourThreat();
  SearchParams params;
  // A zero/uniform network gives the search no prior or value guidance, so
  // discovering that every non-blocking move loses requires visiting each of
  // the ~218 root children ~105 times (White's winning reply sits at row-major
  // index 107). Empirically the threshold is ~23k simulations; 30k gives a
  // comfortable margin. A trained network needs far fewer.
  params.mctsSimulations = 30000;
  params.timeBudgetMs = 60000;  // sims must be the binding constraint, not time
  const Move m = MctsEngine::findBestMove(b, Player::Black, params);
  REQUIRE(m.pos == Position{7, 2});
  MctsEngine::unloadModel();
}

TEST_CASE("MCTS: returns a legal empty move on a mid-game board", "[mcts]") {
  MctsEngine::unloadModel();
  REQUIRE(MctsEngine::loadModel(makeZeroGnn(1, 2)));
  Board b = openFourThreat();
  SearchParams params;
  params.mctsSimulations = 200;
  const Move m = MctsEngine::findBestMove(b, Player::Black, params);
  REQUIRE(b.inBounds(m.pos));
  REQUIRE(b.isEmpty(m.pos));
  MctsEngine::unloadModel();
}

TEST_CASE("MCTS: is deterministic for the same board and params", "[mcts]") {
  MctsEngine::unloadModel();
  REQUIRE(MctsEngine::loadModel(makeZeroGnn(1, 2)));
  Board b = openFourThreat();
  SearchParams params;
  params.mctsSimulations = 400;
  params.timeBudgetMs = 60000;  // the sim count must bind, not the deadline
  const Move a = MctsEngine::findBestMove(b, Player::Black, params);
  const Move c = MctsEngine::findBestMove(b, Player::Black, params);
  REQUIRE(a.pos == c.pos);
  MctsEngine::unloadModel();
}

TEST_CASE("MCTS: returns the only legal move on a near-full board", "[mcts]") {
  MctsEngine::unloadModel();
  REQUIRE(MctsEngine::loadModel(makeZeroGnn(1, 2)));
  Board b;
  for (int r = 0; r < kSize; ++r) {
    for (int c = 0; c < kSize; ++c) {
      if (r == 7 && c == 7) {
        continue;  // the single empty cell
      }
      b.place({Position{r, c}, ((2 * r + c) % 5 < 2) ? Player::Black : Player::White});
    }
  }
  REQUIRE(b.moveCount() == kSize * kSize - 1);
  REQUIRE(WinDetector::anyWinner(b) == Player::None);
  SearchParams params;
  params.mctsSimulations = 200;
  params.timeBudgetMs = 60000;
  const Move m = MctsEngine::findBestMove(b, Player::Black, params);
  REQUIRE(m.pos == Position{7, 7});
  MctsEngine::unloadModel();
}

TEST_CASE("MCTS: respects the budget", "[mcts]") {
  MctsEngine::unloadModel();
  REQUIRE(MctsEngine::loadModel(makeZeroGnn(1, 2)));
  Board b = openFourThreat();
  SearchParams params;
  params.mctsSimulations = 200;
  params.timeBudgetMs = 100;
  const auto start = std::chrono::steady_clock::now();
  const Move m = MctsEngine::findBestMove(b, Player::Black, params);
  const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                           std::chrono::steady_clock::now() - start)
                           .count();
  REQUIRE(b.isEmpty(m.pos));
  REQUIRE(elapsed < 1000);
  MctsEngine::unloadModel();
}
