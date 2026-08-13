#include "ui/BoardWidget.hpp"

#include <QMouseEvent>
#include <QPainter>
#include <QPaintEvent>

#include <array>
#include <cmath>

namespace gomoku {
namespace {

constexpr int kMargin = 24;  // px between board edge and first grid line
constexpr int kCellPx = 36;  // px between grid lines
constexpr int kStoneRadius = 15;

constexpr int kWidgetSize = 2 * kMargin + (kSize - 1) * kCellPx;

}  // namespace

BoardWidget::BoardWidget(QWidget* parent) : QWidget(parent) {
  setMinimumSize(kWidgetSize, kWidgetSize);
  setMouseTracking(false);
}

QSize BoardWidget::sizeHint() const {
  return QSize(kWidgetSize, kWidgetSize);
}

void BoardWidget::setBoard(const Board& board) {
  board_ = board;
}

void BoardWidget::setLastMove(std::optional<Position> lastMove) {
  lastMove_ = lastMove;
}

void BoardWidget::setWinningLine(std::optional<std::array<Position, 5>> line) {
  winningLine_ = line;
}

QPointF BoardWidget::toPixel(Position pos) const {
  return QPointF(kMargin + pos.col * kCellPx, kMargin + pos.row * kCellPx);
}

std::optional<Position> BoardWidget::fromPixel(const QPointF& point) const {
  const int col = static_cast<int>(std::lround((point.x() - kMargin) / kCellPx));
  const int row = static_cast<int>(std::lround((point.y() - kMargin) / kCellPx));
  const Position pos{row, col};
  if (!board_.inBounds(pos)) {
    return std::nullopt;
  }
  // Only accept clicks near an intersection; ignore the wooden margin and
  // the dead space between grid lines.
  const QPointF center = toPixel(pos);
  const double dist = std::hypot(point.x() - center.x(), point.y() - center.y());
  if (dist > kCellPx * 0.6) {
    return std::nullopt;
  }
  return pos;
}

void BoardWidget::paintEvent(QPaintEvent*) {
  QPainter painter(this);
  painter.setRenderHint(QPainter::Antialiasing, true);
  painter.fillRect(rect(), QColor(0xDC, 0xB3, 0x5C));  // wooden board colour

  // Grid lines.
  painter.setPen(QPen(Qt::black, 1));
  for (int i = 0; i < kSize; ++i) {
    painter.drawLine(toPixel(Position{i, 0}), toPixel(Position{i, kSize - 1}));
    painter.drawLine(toPixel(Position{0, i}), toPixel(Position{kSize - 1, i}));
  }

  // Star points (like a Go board).
  painter.setBrush(Qt::black);
  painter.setPen(Qt::NoPen);
  const std::array<Position, 5> stars = {Position{7, 7}, Position{3, 3}, Position{3, 11},
                                         Position{11, 3}, Position{11, 11}};
  for (const Position star : stars) {
    painter.drawEllipse(toPixel(star), 3, 3);
  }

  // Winning line (under the stones).
  if (winningLine_.has_value()) {
    painter.setPen(QPen(QColor(0xC0, 0x20, 0x20), 3));
    painter.drawLine(toPixel((*winningLine_).front()), toPixel((*winningLine_).back()));
  }

  // Stones.
  for (int r = 0; r < kSize; ++r) {
    for (int c = 0; c < kSize; ++c) {
      const Player player = board_.at(Position{r, c});
      if (player == Player::None) {
        continue;
      }
      const QPointF center = toPixel(Position{r, c});
      const bool isBlack = player == Player::Black;
      painter.setPen(QPen(isBlack ? Qt::black : QColor(0x50, 0x50, 0x50), 1));
      painter.setBrush(isBlack ? Qt::black : Qt::white);
      painter.drawEllipse(center, kStoneRadius, kStoneRadius);
    }
  }

  // Last-move marker.
  if (lastMove_.has_value()) {
    painter.setPen(QPen(QColor(0xE0, 0x30, 0x30), 2));
    painter.setBrush(Qt::NoBrush);
    painter.drawEllipse(toPixel(*lastMove_), 5, 5);
  }
}

void BoardWidget::mousePressEvent(QMouseEvent* event) {
  if (event->button() != Qt::LeftButton) {
    return;
  }
  const std::optional<Position> pos = fromPixel(event->position());
  if (!pos.has_value()) {
    return;
  }
  emit cellClicked(pos->row, pos->col);
}

}  // namespace gomoku
