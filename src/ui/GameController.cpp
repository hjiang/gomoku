#include "ui/GameController.hpp"

#include "core/MctsEngine.hpp"
#include "core/WinDetector.hpp"

#include <QMetaObject>

namespace gomoku {

GameController::GameController(QObject* parent) : QObject(parent) {}

GameController::~GameController() {
  ++aiEpoch_;
  if (aiThread_.joinable()) {
    aiThread_.join();
  }
}

void GameController::startNewGame() {
  ++aiEpoch_;  // invalidate any in-flight AI result from the previous game
  if (aiThread_.joinable()) {
    aiThread_.join();
  }
  board_ = Board{};
  state_ = State::WaitingForPlayer;
  emit boardChanged();
  emit statusChanged(QStringLiteral("Your turn (Black)."));
}

void GameController::setDifficulty(int level) {
  if (level < 0 || level > 2) {
    return;
  }
  difficultyLevel_ = level;
}

void GameController::setEngine(int index) {
  // Neural requires a trained model. The UI already disables the option when
  // none is loaded; this guard is defense-in-depth so a programmatic request
  // for Neural can never reach MctsEngine::findBestMove on the worker thread
  // (which throws without a model and would terminate the process).
  engine_ = (index == 1 && MctsEngine::isModelAvailable()) ? EngineKind::Mcts
                                                           : EngineKind::AlphaBeta;
}

void GameController::undoLastMove() {
  if (state_ != State::WaitingForPlayer) {
    return;
  }
  const int count = board_.moveCount();
  if (count == 0) {
    return;
  }
  board_.undo();           // undo the AI move
  if (count >= 2) {
    board_.undo();         // undo the player move that preceded it
  }
  state_ = State::WaitingForPlayer;
  emit boardChanged();
  emit statusChanged(QStringLiteral("Your turn (Black)."));
}

void GameController::onCellClicked(int row, int col) {
  if (state_ != State::WaitingForPlayer) {
    return;
  }
  const Position pos{row, col};
  if (!board_.isEmpty(pos)) {
    return;
  }
  applyMove(Move{pos, humanPlayer()});
}

void GameController::applyMove(Move move) {
  board_.place(move);
  emit boardChanged();

  const Player winner = WinDetector::winnerOf(board_, move.pos);
  if (winner != Player::None) {
    state_ = State::GameOver;
    emit gameEnded(winner);
    emit statusChanged(winner == humanPlayer() ? QStringLiteral("You win! 🎉")
                                               : QStringLiteral("AI wins."));
    return;
  }
  if (board_.isFull()) {
    state_ = State::GameOver;
    emit gameEnded(Player::None);
    emit statusChanged(QStringLiteral("Draw."));
    return;
  }
  if (move.player == humanPlayer()) {
    requestAiMove();
  } else {
    // The AI has replied without ending the game; hand control back to the
    // player. (Without this, state_ would stay AiThinking and freeze input.)
    state_ = State::WaitingForPlayer;
    emit statusChanged(QStringLiteral("Your turn (Black)."));
  }
}

void GameController::requestAiMove() {
  state_ = State::AiThinking;
  emit statusChanged(QStringLiteral("AI is thinking…"));
  emit boardChanged();

  const Board snapshot = board_;
  SearchParams params = SearchEngine::difficulty(difficultyLevel_);
  params.engine = engine_;
  const int epoch = ++aiEpoch_;
  if (aiThread_.joinable()) {
    aiThread_.join();
  }

  aiThread_ = std::thread([this, snapshot, params, epoch] {
    const Move move = SearchEngine::findBestMove(snapshot, aiPlayer(), params);
    QMetaObject::invokeMethod(this, [this, epoch, move] { finishAiTurn(epoch, move); },
                              Qt::QueuedConnection);
  });
}

void GameController::finishAiTurn(int epoch, Move move) {
  if (epoch != aiEpoch_) {
    return;  // a newer search or a new game has superseded this result
  }
  if (state_ != State::AiThinking) {
    return;
  }
  applyMove(move);
}

}  // namespace gomoku
