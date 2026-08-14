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
  // Loads the .gnn model into MctsEngine so the Neural engine becomes
  // available. Lookup order: GOMOKU_MODEL_PATH (explicit override), then a
  // bundled model next to the binary or in the installed share/ dir. Call
  // before the engine combo is built.
  void loadModel();
  // Reads `path` and parses it as a .gnn model; returns true on success.
  // A failed load leaves any previously loaded model unchanged.
  [[nodiscard]] static bool tryLoadModelFile(const QString& path);

  BoardWidget* boardWidget_ = nullptr;
  GameController* controller_ = nullptr;
  QComboBox* difficulty_ = nullptr;
  QComboBox* engine_ = nullptr;
  QLabel* status_ = nullptr;
};

}  // namespace gomoku
