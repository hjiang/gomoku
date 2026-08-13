#pragma once

#include "core/Board.hpp"
#include "core/SearchEngine.hpp"

#include <QObject>

#include <thread>

namespace gomoku {

// State machine for a player-vs-AI game. Runs the AI search on a worker
// thread and applies the result back on the UI thread via a queued call.
class GameController : public QObject {
  Q_OBJECT

 public:
  enum class State { WaitingForPlayer, AiThinking, GameOver };

  explicit GameController(QObject* parent = nullptr);
  ~GameController() override;

  [[nodiscard]] State state() const { return state_; }
  [[nodiscard]] const Board& board() const { return board_; }
  [[nodiscard]] Player humanPlayer() const { return Player::Black; }
  [[nodiscard]] Player aiPlayer() const { return Player::White; }
  [[nodiscard]] EngineKind engine() const { return engine_; }

  void startNewGame();
  void setDifficulty(int level);  // 0 = Easy, 1 = Medium, 2 = Hard
  // 0 = AlphaBeta (Classic), 1 = Mcts (Neural). Neural is rejected (stays
  // Classic) when no trained model is loaded.
  void setEngine(int index);
  void undoLastMove();

 signals:
  void boardChanged();
  void statusChanged(const QString& text);
  void gameEnded(Player winner);  // Player::None means draw

 public slots:
  void onCellClicked(int row, int col);

 private:
  void applyMove(Move move);
  void requestAiMove();
  void finishAiTurn(int epoch, Move move);

  Board board_;
  State state_ = State::WaitingForPlayer;
  int difficultyLevel_ = 2;  // 0 = Easy, 1 = Medium, 2 = Hard (default)
  EngineKind engine_ = EngineKind::AlphaBeta;  // Classic by default
  std::thread aiThread_;
  // Bumped on the UI thread for every new search / new game; a queued AI
  // result is applied only if its captured epoch still matches.
  int aiEpoch_ = 0;
};

}  // namespace gomoku
