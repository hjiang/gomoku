#include "ui/MainWindow.hpp"

#include "core/WinDetector.hpp"
#include "ui/BoardWidget.hpp"
#include "ui/GameController.hpp"

#include <QComboBox>
#include <QLabel>
#include <QStatusBar>
#include <QToolBar>

namespace gomoku {

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent) {
  setWindowTitle(QStringLiteral("Gomoku — Player vs AI"));

  boardWidget_ = new BoardWidget(this);
  setCentralWidget(boardWidget_);

  controller_ = new GameController(this);

  QToolBar* toolbar = addToolBar(QStringLiteral("Game"));
  toolbar->setMovable(false);
  toolbar->addAction(QStringLiteral("New Game"), this, &MainWindow::onNewGame);
  toolbar->addAction(QStringLiteral("Undo"), this, &MainWindow::onUndo);
  toolbar->addSeparator();
  toolbar->addWidget(new QLabel(QStringLiteral(" Difficulty: ")));
  difficulty_ = new QComboBox(toolbar);
  difficulty_->addItems({QStringLiteral("Easy"), QStringLiteral("Medium"),
                         QStringLiteral("Hard")});
  difficulty_->setCurrentIndex(1);
  toolbar->addWidget(difficulty_);

  status_ = new QLabel(QStringLiteral("Your turn (Black)."));
  statusBar()->addWidget(status_);

  connect(boardWidget_, &BoardWidget::cellClicked, controller_, &GameController::onCellClicked);
  connect(controller_, &GameController::boardChanged, this, &MainWindow::onBoardChanged);
  connect(controller_, &GameController::statusChanged, this, &MainWindow::onStatusChanged);
  connect(difficulty_, &QComboBox::currentIndexChanged, controller_,
          &GameController::setDifficulty);

  resize(560, 560);
  controller_->startNewGame();
}

void MainWindow::onNewGame() {
  controller_->startNewGame();
}

void MainWindow::onUndo() {
  controller_->undoLastMove();
}

void MainWindow::onBoardChanged() {
  const Board& board = controller_->board();
  boardWidget_->setBoard(board);

  const auto last = board.lastMove();
  boardWidget_->setLastMove(last ? std::optional<Position>(last->pos) : std::nullopt);

  // Recompute the winning line from the board itself, so board rendering is
  // self-contained (no reliance on a separate game-over notification).
  std::optional<std::array<Position, 5>> line;
  if (last.has_value()) {
    line = WinDetector::findWinningLine(board, last->pos);
  }
  boardWidget_->setWinningLine(line);
  boardWidget_->update();
}

void MainWindow::onStatusChanged(const QString& text) {
  status_->setText(text);
}



}  // namespace gomoku
