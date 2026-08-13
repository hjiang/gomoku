#include "ui/GameController.hpp"

#include <QtTest/QtTest>

using namespace gomoku;

// Regression test for the state machine: after the AI replies with a
// non-winning move, control must return to the player (state must leave
// AiThinking). This exercises the real worker thread and queued result
// delivery, which the pure core tests cannot cover.
class GameControllerTest : public QObject {
  Q_OBJECT

 private slots:
  void playerMoveReturnsControlAfterAiReply();
};

void GameControllerTest::playerMoveReturnsControlAfterAiReply() {
  GameController controller;
  controller.setDifficulty(0);  // Easy: depth 2, short budget
  controller.startNewGame();

  QVERIFY(controller.state() == GameController::State::WaitingForPlayer);

  // Player places the first stone.
  controller.onCellClicked(7, 7);
  QVERIFY(controller.state() == GameController::State::AiThinking);
  QCOMPARE(controller.board().moveCount(), 1);

  // Wait for the AI result (delivered via a queued call) and verify the state
  // machine returns to the player's turn instead of freezing in AiThinking.
  QTRY_VERIFY_WITH_TIMEOUT(
      controller.state() == GameController::State::WaitingForPlayer, 5000);
  QCOMPARE(controller.board().moveCount(), 2);
}

QTEST_GUILESS_MAIN(GameControllerTest)
#include "test_controller.moc"
