// Headless self-play (RL) training-data generator: loads a trained .gnn model
// and plays MCTS against itself (root Dirichlet noise), writing the games as a
// GameRecord stream. Developer tooling only — never installed, never part of
// the shipped game binary.

#include "core/MctsEngine.hpp"
#include "core/SearchEngine.hpp"
#include "core/SelfPlay.hpp"

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
  int games = 100;
  int sims = 800;
  int timeMs = 5000;
  std::uint32_t seed = 1;
  std::string out = "selfplay.rec";
};

void usage() {
  std::cerr << "usage: gomoku-selfplay --model FILE.gnn --games N --sims M --time-ms T "
               "--seed S --out FILE\n";
}

bool parseArgs(int argc, char** argv, Options& opts) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--model" || arg == "--games" || arg == "--sims" || arg == "--time-ms" ||
        arg == "--seed" || arg == "--out") {
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
      } else if (arg == "--time-ms") {
        opts.timeMs = std::atoi(value.c_str());
      } else if (arg == "--seed") {
        opts.seed = static_cast<std::uint32_t>(std::strtoul(value.c_str(), nullptr, 10));
      } else {
        opts.out = value;
      }
    } else {
      std::cerr << "unknown argument: " << arg << '\n';
      return false;
    }
  }
  return !opts.model.empty() && opts.games >= 1 && opts.sims >= 1 && opts.timeMs >= 0;
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

  SearchParams params;
  params.mctsSimulations = opts.sims;
  params.timeBudgetMs = opts.timeMs;

  std::vector<GameRecord> games;
  games.reserve(static_cast<std::size_t>(opts.games));
  for (int i = 0; i < opts.games; ++i) {
    // A distinct nonzero per-game seed so the Dirichlet noise explores
    // different lines across games.
    std::uint32_t gameSeed = opts.seed + static_cast<std::uint32_t>(i) + 1;
    if (gameSeed == 0) {
      gameSeed = 1;  // a wrapped-to-0 seed would silently repeat a game
    }
    games.push_back(generateSelfPlayGame(gameSeed, params));
    std::cerr << "game " << (i + 1) << "/" << opts.games << ": "
              << games.back().positions.size() << " positions\n";
  }
  MctsEngine::unloadModel();

  const std::vector<std::uint8_t> bytes = encodeStream(games);
  std::ofstream out(opts.out, std::ios::binary);
  if (!out) {
    std::cerr << "cannot open " << opts.out << '\n';
    return 1;
  }
  if (!bytes.empty()) {
    out.write(reinterpret_cast<const char*>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
  }
  if (!out) {
    std::cerr << "failed to write " << opts.out << '\n';
    return 1;
  }
  std::cerr << "wrote " << bytes.size() << " bytes (" << games.size() << " games) to "
            << opts.out << '\n';
  return 0;
}
