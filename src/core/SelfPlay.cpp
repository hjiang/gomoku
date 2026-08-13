#include "core/SelfPlay.hpp"

#include "core/AlphaBetaEngine.hpp"
#include "core/BoardEncoder.hpp"
#include "core/MctsEngine.hpp"
#include "core/WinDetector.hpp"

#include <array>
#include <cstdint>

namespace gomoku {
namespace {

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
  // different seeds explore different lines (see the plan's risk table).
  SearchParams search = params;
  search.seed = seed;

  Board board;
  Player toMove = Player::Black;
  GameRecord game;

  for (;;) {
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

    const Move move = AlphaBetaEngine::findBestMove(board, toMove, search);
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
