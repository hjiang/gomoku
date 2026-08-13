#include "ui/MainWindow.hpp"

#include "core/MctsEngine.hpp"
#include "core/WinDetector.hpp"
#include "ui/BoardWidget.hpp"
#include "ui/GameController.hpp"

#include <QComboBox>
#include <QFile>
#include <QLabel>
#include <QStandardItem>
#include <QStandardItemModel>
#include <QStatusBar>
#include <QToolBar>

#include <cstdint>
#include <cstdlib>
#include <span>

namespace gomoku {

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent) {
  setWindowTitle(QStringLiteral("Gomoku — Player vs AI"));

  boardWidget_ = new BoardWidget(this);
  setCentralWidget(boardWidget_);

  loadModelFromEnv();  // enables the Neural engine when a model file exists

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
  difficulty_->setCurrentIndex(2);
  toolbar->addWidget(difficulty_);
  toolbar->addSeparator();
  toolbar->addWidget(new QLabel(QStringLiteral(" Engine: ")));
  engine_ = new QComboBox(toolbar);
  {
    auto* model = new QStandardItemModel(engine_);
    model->appendRow(new QStandardItem(QStringLiteral("Classic")));
    auto* neuralItem = new QStandardItem(QStringLiteral("Neural"));
    if (!MctsEngine::isModelAvailable()) {
      neuralItem->setEnabled(false);
      neuralItem->setToolTip(
          QStringLiteral("Neural engine requires a trained model (none loaded)."));
    }
    model->appendRow(neuralItem);
    engine_->setModel(model);
  }
  engine_->setCurrentIndex(0);
  toolbar->addWidget(engine_);

  status_ = new QLabel(QStringLiteral("Your turn (Black)."));
  statusBar()->addWidget(status_);

  connect(boardWidget_, &BoardWidget::cellClicked, controller_, &GameController::onCellClicked);
  connect(controller_, &GameController::boardChanged, this, &MainWindow::onBoardChanged);
  connect(controller_, &GameController::statusChanged, this, &MainWindow::onStatusChanged);
  connect(difficulty_, &QComboBox::currentIndexChanged, controller_,
          &GameController::setDifficulty);
  connect(engine_, &QComboBox::currentIndexChanged, controller_, &GameController::setEngine);

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

void MainWindow::loadModelFromEnv() {
  const char* path = std::getenv("GOMOKU_MODEL_PATH");
  if (path == nullptr || *path == '\0') {
    return;
  }
  QFile file(QString::fromUtf8(path));
  if (!file.open(QIODevice::ReadOnly)) {
    return;
  }
  const QByteArray data = file.readAll();
  file.close();
  if (data.isEmpty()) {
    return;
  }
  const auto* bytes = reinterpret_cast<const std::uint8_t*>(data.constData());
  static_cast<void>(MctsEngine::loadModel(
      std::span<const std::uint8_t>(bytes, static_cast<std::size_t>(data.size()))));
}

}  // namespace gomoku
