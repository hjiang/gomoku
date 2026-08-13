#include "ui/MainWindow.hpp"

#include <QApplication>

int main(int argc, char** argv) {
  QApplication app(argc, argv);
  gomoku::MainWindow window;
  window.show();
  return app.exec();
}
