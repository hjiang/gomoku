#pragma once

#include "core/types.hpp"

#include <QMainWindow>

#include <array>
#include <optional>

class QComboBox;
class QLabel;

namespace gomoku {

class BoardWidget;
class GameController;

class MainWindow : public QMainWindow {
  Q_OBJECT

 public:
  explicit MainWindow(QWidget* parent = nullptr);

 private slots:
  void onNewGame();
  void onUndo();

 private:
  void onBoardChanged();
  void onStatusChanged(const QString& text);
  // Reads GOMOKU_MODEL_PATH (if set) and loads the .gnn model into
  // MctsEngine so the Neural engine becomes available. Call before the
  // engine combo is built.
  void loadModelFromEnv();

  BoardWidget* boardWidget_ = nullptr;
  GameController* controller_ = nullptr;
  QComboBox* difficulty_ = nullptr;
  QComboBox* engine_ = nullptr;
  QLabel* status_ = nullptr;
};

}  // namespace gomoku
