#pragma once

#include "core/GameRecord.hpp"
#include "core/SearchEngine.hpp"

#include <cstdint>

namespace gomoku {

// Headless game generation for training data (Stage 3). Both generators play a
// full game and record every position in the shared GameRecord format: the
// network input planes, a policy target, and the outcome value. Qt-free, so
// the generators are hermetic-testable without the tools or the filesystem.
//
// Bootstrap: AlphaBetaEngine vs itself; the policy target is the alpha-beta
// move, label-smoothed so the policy head is not brittle, and the value target
// is the game outcome.
// Self-play: MctsEngine vs itself via MctsEngine::selfPlay (root Dirichlet
// noise); the policy target is the root visit-count distribution.
[[nodiscard]] GameRecord generateBootstrapGame(std::uint32_t seed, const SearchParams& params);

// Pre: MctsEngine::isModelAvailable() (selfPlay throws otherwise).
[[nodiscard]] GameRecord generateSelfPlayGame(std::uint32_t seed, const SearchParams& params);

}  // namespace gomoku
