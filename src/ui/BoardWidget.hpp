#pragma once

#include "core/Board.hpp"

#include <QWidget>

#include <array>
#include <optional>

class QMouseEvent;

namespace gomoku {

// Custom-painted Gomoku board: grid, stones, last-move marker, winning line.
class BoardWidget : public QWidget {
  Q_OBJECT

 public:
  explicit BoardWidget(QWidget* parent = nullptr);

  void setBoard(const Board& board);
  void setLastMove(std::optional<Position> lastMove);
  void setWinningLine(std::optional<std::array<Position, 5>> line);

 signals:
  void cellClicked(int row, int col);

 protected:
  void paintEvent(QPaintEvent* event) override;
  void mousePressEvent(QMouseEvent* event) override;
  QSize sizeHint() const override;

 private:
  [[nodiscard]] QPointF toPixel(Position pos) const;
  [[nodiscard]] std::optional<Position> fromPixel(const QPointF& point) const;

  Board board_;
  std::optional<Position> lastMove_;
  std::optional<std::array<Position, 5>> winningLine_;
};

}  // namespace gomoku
