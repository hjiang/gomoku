// Headless supervised (bootstrap) training-data generator: plays AlphaBeta
// against itself and writes the games as a GameRecord stream. Developer
// tooling only — never installed, never part of the shipped game binary.

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
  int games = 100;
  int depth = 4;
  int timeMs = 1000;
  std::uint32_t seed = 1;
  std::string out = "bootstrap.rec";
};

void usage() {
  std::cerr << "usage: gomoku-bootstrap --games N --depth D --time-ms T --seed S --out FILE\n";
}

bool parseArgs(int argc, char** argv, Options& opts) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--games" || arg == "--depth" || arg == "--time-ms" || arg == "--seed" ||
        arg == "--out") {
      if (i + 1 >= argc) {
        std::cerr << "missing value for " << arg << '\n';
        return false;
      }
      const std::string value = argv[++i];
      if (arg == "--games") {
        opts.games = std::atoi(value.c_str());
      } else if (arg == "--depth") {
        opts.depth = std::atoi(value.c_str());
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
  return opts.games >= 1 && opts.depth >= 1 && opts.timeMs >= 0;
}

}  // namespace

int main(int argc, char** argv) {
  Options opts;
  if (!parseArgs(argc, argv, opts)) {
    usage();
    return 1;
  }

  SearchParams params;
  params.maxDepth = opts.depth;
  params.timeBudgetMs = opts.timeMs;

  std::vector<GameRecord> games;
  games.reserve(static_cast<std::size_t>(opts.games));
  for (int i = 0; i < opts.games; ++i) {
    // A distinct nonzero per-game seed so successive games differ.
    std::uint32_t gameSeed = opts.seed + static_cast<std::uint32_t>(i) + 1;
    if (gameSeed == 0) {
      gameSeed = 1;  // a wrapped-to-0 seed would silently repeat a game
    }
    games.push_back(generateBootstrapGame(gameSeed, params));
    std::cerr << "game " << (i + 1) << "/" << opts.games << ": "
              << games.back().positions.size() << " positions\n";
  }

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
