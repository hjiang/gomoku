#include "core/MctsEngine.hpp"

#include "core/BoardEncoder.hpp"
#include "core/NeuralNet.hpp"
#include "core/TensorOps.hpp"
#include "core/Weights.hpp"
#include "core/WinDetector.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace gomoku {
namespace {

// PUCT exploration constant.
constexpr float kPuct = 2.0f;

// A position in the MCTS tree. Q and W are always from the perspective of
// `toMove` (the player to move at this node): +1 means toMove wins.
struct Node {
  Board board;
  Player toMove = Player::None;
  bool terminal = false;
  float terminalValue = 0.0f;  // from toMove perspective; valid only if terminal
  bool expanded = false;
  std::vector<std::unique_ptr<Node>> children;
  float prior = 0.0f;  // policy prior of the move that produced this node
  float Q = 0.0f;
  float W = 0.0f;
  int N = 0;
  Move moveToHere{};  // the move that produced this node (root: default)
};

// Child value from the parent's perspective is -child.Q (child.toMove is the
// opponent). UCT = Q(s,a) + cPuct * P(s,a) * sqrt(N(s)) / (1 + N(s,a)).
float uct(const Node& parent, const Node& child) {
  return -child.Q + kPuct * child.prior * std::sqrt(static_cast<float>(parent.N)) /
                        (1.0f + static_cast<float>(child.N));
}

// Expands `node`: evaluates the network, masks illegal cells, creates one
// child per empty cell (terminal children for wins/draws). Returns the
// network value v from `node.toMove`'s perspective.
float expand(Node& node, const NeuralNet& model) {
  if (node.board.isFull()) {
    node.terminal = true;
    node.terminalValue = 0.0f;
    return 0.0f;
  }

  const auto enc = BoardEncoder::encode(node.board, node.toMove);
  const NeuralNet::Output out = model.evaluate(enc);

  std::array<float, kSize * kSize> masked = out.policyLogits;
  for (int r = 0; r < kSize; ++r) {
    for (int c = 0; c < kSize; ++c) {
      if (!node.board.isEmpty(Position{r, c})) {
        masked[static_cast<std::size_t>(r) * kSize + c] =
            -std::numeric_limits<float>::infinity();
      }
    }
  }
  const std::vector<float> probs = softmax(masked);

  for (int r = 0; r < kSize; ++r) {
    for (int c = 0; c < kSize; ++c) {
      const Position pos{r, c};
      if (!node.board.isEmpty(pos)) {
        continue;
      }
      auto child = std::make_unique<Node>();
      child->moveToHere = Move{pos, node.toMove};
      child->board = node.board;
      child->board.place(child->moveToHere);
      child->toMove = opponent(node.toMove);
      if (WinDetector::winnerOf(child->board, pos) == node.toMove) {
        child->terminal = true;
        child->terminalValue = -1.0f;  // child->toMove (opponent) loses
      } else if (child->board.isFull()) {
        child->terminal = true;
        child->terminalValue = 0.0f;
      }
      child->prior = probs[static_cast<std::size_t>(r) * kSize + c];
      node.children.push_back(std::move(child));
    }
  }
  node.expanded = true;
  return out.value;
}

// Descends from the root along the PUCT-maximizing path to an unexpanded
// leaf (or a terminal node). Returns the path (root first, leaf last) and the
// leaf's value from the leaf's own toMove perspective.
struct SearchResult {
  std::vector<Node*> path;
  float value;
};

SearchResult select(Node& root, const NeuralNet& model) {
  std::vector<Node*> path;
  path.push_back(&root);
  Node* cur = &root;
  while (cur->expanded && !cur->terminal) {
    assert(!cur->children.empty());
    Node* best = cur->children.front().get();
    float bestScore = uct(*cur, *best);
    for (std::size_t i = 1; i < cur->children.size(); ++i) {
      Node* child = cur->children[i].get();
      const float s = uct(*cur, *child);
      if (s > bestScore) {
        bestScore = s;
        best = child;
      }
    }
    cur = best;
    path.push_back(cur);
  }
  if (cur->terminal) {
    return {std::move(path), cur->terminalValue};
  }
  return {std::move(path), expand(*cur, model)};
}

// Propagates the leaf value up the path, negating at each level so every
// node's Q/W stay in that node's own toMove perspective.
void backup(const std::vector<Node*>& path, float value) {
  float cur = value;
  for (auto it = path.rbegin(); it != path.rend(); ++it) {
    Node* node = *it;
    ++node->N;
    node->W += cur;
    node->Q = node->W / static_cast<float>(node->N);
    cur = -cur;
  }
}

}  // namespace

// The loaded network. Set once at startup (UI thread) before any search runs
// on the worker thread, so no locking is needed.
std::optional<NeuralNet>& modelStorage() {
  static std::optional<NeuralNet> model;
  return model;
}

bool MctsEngine::isModelAvailable() {
  return modelStorage().has_value();
}

bool MctsEngine::loadModel(std::span<const std::uint8_t> bytes) {
  const std::optional<Weights> weights = Weights::parse(bytes);
  if (!weights) {
    return false;  // leave any previously loaded model unchanged
  }
  modelStorage().emplace(std::move(*weights));
  return true;
}

void MctsEngine::unloadModel() {
  modelStorage().reset();
}

Move MctsEngine::findBestMove(const Board& board, Player player, const SearchParams& params) {
  if (!modelStorage().has_value()) {
    throw std::runtime_error("MCTS engine requires trained weights (none loaded)");
  }
  assert(WinDetector::anyWinner(board) == Player::None);
  assert(board.moveCount() < kSize * kSize);

  Node root;
  root.board = board;
  root.toMove = player;

  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(params.timeBudgetMs);
  const int sims = std::max(1, params.mctsSimulations);

  for (int i = 0; i < sims; ++i) {
    if (i > 0 && std::chrono::steady_clock::now() >= deadline) {
      break;
    }
    const SearchResult result = select(root, *modelStorage());
    backup(result.path, result.value);
  }

  assert(!root.children.empty());
  std::size_t bestIdx = 0;
  for (std::size_t i = 1; i < root.children.size(); ++i) {
    if (root.children[i]->N > root.children[bestIdx]->N) {
      bestIdx = i;
    }
  }
  return root.children[bestIdx]->moveToHere;
}

}  // namespace gomoku
