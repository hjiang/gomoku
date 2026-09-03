#include "core/SelfPlay.hpp"

#include "core/AlphaBetaEngine.hpp"
#include "core/BoardEncoder.hpp"
#include "core/MctsEngine.hpp"
#include "core/WinDetector.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <random>
#include <vector>

namespace gomoku {
namespace {

// Uniform-random empty cell within `radius` of the board center (7,7), or
// nullopt when the region is fully occupied. Used for the opening-jitter plies
// of bootstrap games: randomizing the first few moves makes White win a real
// share of games, balancing the value targets (without it, Black wins ~99% of
// teacher-vs-teacher games and the value head degenerates into a parity
// detector). The policy target for a jittered ply is the label-smoothed target
// on whichever move was actually played, exactly like a teacher move.
std::optional<Position> pickRandomOpeningMove(const Board& board, std::mt19937_64& rng,
                                              int radius) {
  const int center = kSize / 2;
  std::vector<Position> candidates;
  const int side = 2 * radius + 1;
  candidates.reserve(static_cast<std::size_t>(side) * side);
  for (int r = center - radius; r <= center + radius; ++r) {
    for (int c = center - radius; c <= center + radius; ++c) {
      if (r < 0 || r >= kSize || c < 0 || c >= kSize) {
        continue;
      }
      if (board.isEmpty(Position{r, c})) {
        candidates.emplace_back(Position{r, c});
      }
    }
  }
  if (candidates.empty()) {
    return std::nullopt;
  }
  std::uniform_int_distribution<std::size_t> dist(0, candidates.size() - 1);
  return candidates[dist(rng)];
}

// Back-fills every position's value from the game outcome. Records always
// start with Black to move and alternate strictly, so the player to move at
// position i is Black exactly when i is even.
void fillValues(GameRecord& game, Player winner) {
  for (std::size_t i = 0; i < game.positions.size(); ++i) {
    const Player toMove = (i % 2 == 0) ? Player::Black : Player::White;
    float value = 0.0f;
    if (winner == Player::Black) {
      value = toMove == Player::Black ? 1.0f : -1.0f;
    } else if (winner == Player::White) {
      value = toMove == Player::White ? 1.0f : -1.0f;
    }
    game.positions[i].value = value;
  }
}

}  // namespace

GameRecord generateBootstrapGame(std::uint32_t seed, const SearchParams& params) {
  // The per-game seed drives AlphaBeta's root-candidate shuffle so games with
  // different seeds explore different lines (see the plan's risk table). The
  // same seed also drives the opening-jitter RNG (when params.openingJitter >
  // 0), so the whole game stays deterministic per seed.
  SearchParams search = params;
  search.seed = seed;

  // Seeded from the game seed (cast to 64 bits for the mt19937_64 state).
  std::mt19937_64 rng(static_cast<std::uint64_t>(seed));

  Board board;
  Player toMove = Player::Black;
  GameRecord game;

  for (int ply = 0;; ++ply) {
    std::array<bool, kSize * kSize> legal{};
    for (int r = 0; r < kSize; ++r) {
      for (int c = 0; c < kSize; ++c) {
        if (board.isEmpty(Position{r, c})) {
          legal[static_cast<std::size_t>(r) * kSize + static_cast<std::size_t>(c)] = true;
        }
      }
    }

    PositionRecord rec;
    rec.planes = BoardEncoder::encode(board, toMove);

    // The first openingJitter plies are uniform-random within the center
    // region (see pickRandomOpeningMove). If the region has no empty cell,
    // fall back to the alpha-beta teacher move.
    Move move;
    if (ply < search.openingJitter) {
      const std::optional<Position> random =
          pickRandomOpeningMove(board, rng, search.openingRadius);
      move = random.has_value() ? Move{*random, toMove}
                                : AlphaBetaEngine::findBestMove(board, toMove, search);
    } else {
      move = AlphaBetaEngine::findBestMove(board, toMove, search);
    }
    const int moveIdx = move.pos.row * kSize + move.pos.col;
    rec.policy = labelSmoothedPolicy(moveIdx, legal);
    game.positions.push_back(std::move(rec));

    board.place(move);
    const Player winner = WinDetector::winnerOf(board, move.pos);
    if (winner != Player::None) {
      fillValues(game, winner);
      return game;
    }
    if (board.isFull()) {
      fillValues(game, Player::None);
      return game;
    }
    toMove = opponent(toMove);
  }
}

GameRecord generateSelfPlayGame(std::uint32_t seed, const SearchParams& params) {
  SearchParams search = params;
  search.seed = seed;  // seeds the root Dirichlet noise in selfPlay

  Board board;
  Player toMove = Player::Black;
  GameRecord game;

  for (;;) {
    PositionRecord rec;
    rec.planes = BoardEncoder::encode(board, toMove);

    const MctsEngine::SelfPlayResult result = MctsEngine::selfPlay(board, toMove, search);
    rec.policy = result.policy;
    game.positions.push_back(std::move(rec));

    board.place(result.move);
    const Player winner = WinDetector::winnerOf(board, result.move.pos);
    if (winner != Player::None) {
      fillValues(game, winner);
      return game;
    }
    if (board.isFull()) {
      fillValues(game, Player::None);
      return game;
    }
    toMove = opponent(toMove);
  }
}

}  // namespace gomoku
