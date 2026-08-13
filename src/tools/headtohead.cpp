// Headless head-to-head gate: plays the neural (MCTS) engine against the
// classic (alpha-beta) engine and reports the score. The classic engine runs
// at Classic-Hard strength by default (depth 6, 2000 ms); the MCTS engine's
// simulation budget is configurable (Classic Hard uses 1600). Developer
// tooling only — never installed, never part of the shipped game binary.

#include "core/AlphaBetaEngine.hpp"
#include "core/Board.hpp"
#include "core/MctsEngine.hpp"
#include "core/SearchEngine.hpp"
#include "core/WinDetector.hpp"

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace gomoku;

namespace {

struct Options {
  std::string model;
  int games = 10;
  int sims = 1600;   // MCTS simulation budget (Classic Hard default)
  int depth = 6;     // alpha-beta depth (Classic Hard default)
  int timeMs = 2000; // soft deadline for both engines (Classic Hard default)
  std::uint32_t seed = 1;
};

void usage() {
  std::cerr << "usage: gomoku-headtohead --model FILE.gnn --games N --sims M "
               "--depth D --time-ms T --seed S\n";
}

bool parseArgs(int argc, char** argv, Options& opts) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--model" || arg == "--games" || arg == "--sims" || arg == "--depth" ||
        arg == "--time-ms" || arg == "--seed") {
      if (i + 1 >= argc) {
        std::cerr << "missing value for " << arg << '\n';
        return false;
      }
      const std::string value = argv[++i];
      if (arg == "--model") {
        opts.model = value;
      } else if (arg == "--games") {
        opts.games = std::atoi(value.c_str());
      } else if (arg == "--sims") {
        opts.sims = std::atoi(value.c_str());
      } else if (arg == "--depth") {
        opts.depth = std::atoi(value.c_str());
      } else if (arg == "--time-ms") {
        opts.timeMs = std::atoi(value.c_str());
      } else {
        opts.seed = static_cast<std::uint32_t>(std::strtoul(value.c_str(), nullptr, 10));
      }
    } else {
      std::cerr << "unknown argument: " << arg << '\n';
      return false;
    }
  }
  return !opts.model.empty() && opts.games >= 1 && opts.sims >= 1 && opts.depth >= 1 &&
         opts.timeMs >= 0;
}

bool readFile(const std::string& path, std::vector<std::uint8_t>& bytes) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return false;
  }
  in.seekg(0, std::ios::end);
  const std::streamoff size = in.tellg();
  if (size < 0) {
    return false;
  }
  in.seekg(0, std::ios::beg);
  bytes.resize(static_cast<std::size_t>(size));
  if (size > 0) {
    in.read(reinterpret_cast<char*>(bytes.data()), size);
  }
  return static_cast<bool>(in);
}

// Plays one full game. `mctsIsBlack` picks the colors (the MCTS engine plays
// that color; alpha-beta plays the other). `ab` carries the per-game seed so
// alpha-beta's root-candidate shuffle explores different lines. Returns the
// winner (Black or White) or Player::None for a draw.
Player playGame(bool mctsIsBlack, const SearchParams& mcts, const SearchParams& ab) {
  const Player mctsColor = mctsIsBlack ? Player::Black : Player::White;

  Board board;
  Player toMove = Player::Black;
  for (;;) {
    const Move move = (toMove == mctsColor)
                          ? MctsEngine::findBestMove(board, toMove, mcts)
                          : AlphaBetaEngine::findBestMove(board, toMove, ab);
    board.place(move);
    const Player winner = WinDetector::winnerOf(board, move.pos);
    if (winner != Player::None || board.isFull()) {
      return winner;
    }
    toMove = opponent(toMove);
  }
}

}  // namespace

int main(int argc, char** argv) {
  Options opts;
  if (!parseArgs(argc, argv, opts)) {
    usage();
    return 1;
  }

  std::vector<std::uint8_t> modelBytes;
  if (!readFile(opts.model, modelBytes)) {
    std::cerr << "cannot read model file " << opts.model << '\n';
    return 1;
  }
  if (!MctsEngine::loadModel(modelBytes)) {
    std::cerr << "failed to load model from " << opts.model << '\n';
    return 1;
  }

  SearchParams mcts;
  mcts.mctsSimulations = opts.sims;
  mcts.timeBudgetMs = opts.timeMs;

  SearchParams ab;
  ab.maxDepth = opts.depth;
  ab.timeBudgetMs = opts.timeMs;

  int mctsWins = 0;
  int abWins = 0;
  int draws = 0;
  int mctsBlackWins = 0;
  int mctsWhiteWins = 0;
  int abBlackWins = 0;
  int abWhiteWins = 0;

  for (int i = 0; i < opts.games; ++i) {
    // Alternate colors so neither engine always moves first; MCTS is Black on
    // odd games. A distinct nonzero seed drives alpha-beta's root shuffle.
    const bool mctsIsBlack = (i % 2 == 1);
    std::uint32_t gameSeed = opts.seed + static_cast<std::uint32_t>(i) + 1;
    if (gameSeed == 0) {
      gameSeed = 1;
    }
    SearchParams abGame = ab;
    abGame.seed = gameSeed;

    const Player winner = playGame(mctsIsBlack, mcts, abGame);
    const Player mctsColor = mctsIsBlack ? Player::Black : Player::White;
    if (winner == Player::None) {
      ++draws;
      std::cerr << "game " << (i + 1) << "/" << opts.games << ": draw\n";
    } else if (winner == mctsColor) {
      ++mctsWins;
      if (winner == Player::Black) {
        ++mctsBlackWins;
      } else {
        ++mctsWhiteWins;
      }
      std::cerr << "game " << (i + 1) << "/" << opts.games << ": mcts wins ("
                << (winner == Player::Black ? "black" : "white") << ")\n";
    } else {
      ++abWins;
      if (winner == Player::Black) {
        ++abBlackWins;
      } else {
        ++abWhiteWins;
      }
      std::cerr << "game " << (i + 1) << "/" << opts.games << ": alpha-beta wins ("
                << (winner == Player::Black ? "black" : "white") << ")\n";
    }
  }
  MctsEngine::unloadModel();

  std::cout << "games " << opts.games << '\n';
  std::cout << "mcts_wins " << mctsWins << '\n';
  std::cout << "ab_wins " << abWins << '\n';
  std::cout << "draws " << draws << '\n';
  std::cout << "mcts_black_wins " << mctsBlackWins << '\n';
  std::cout << "mcts_white_wins " << mctsWhiteWins << '\n';
  std::cout << "ab_black_wins " << abBlackWins << '\n';
  std::cout << "ab_white_wins " << abWhiteWins << '\n';
  return 0;
}
