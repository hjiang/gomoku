cmake_minimum_required(VERSION 3.21)
project(gomoku VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

# --- Warning policy (NFR6) ---------------------------------------------------
option(GOMOKU_WERROR "Treat warnings as errors" ON)
set(GOMOKU_WARN_FLAGS -Wall -Wextra)
if(GOMOKU_WERROR)
  list(APPEND GOMOKU_WARN_FLAGS -Werror)
endif()

# --- Core engine: Qt-free static library --------------------------------------
add_library(gomoku_core STATIC
  src/core/Board.cpp
  src/core/WinDetector.cpp
  src/core/PatternTable.cpp
  src/core/Evaluator.cpp
  src/core/SearchEngine.cpp
  src/core/AlphaBetaEngine.cpp
  src/core/MctsEngine.cpp
  src/core/BoardEncoder.cpp
  src/core/Weights.cpp
  src/core/NeuralNet.cpp
  src/core/GameRecord.cpp
  src/core/SelfPlay.cpp
)
target_include_directories(gomoku_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_options(gomoku_core PRIVATE ${GOMOKU_WARN_FLAGS})

# --- Unit tests (Catch2) ------------------------------------------------------
enable_testing()
find_package(Catch2 3 REQUIRED)
include(Catch)

add_executable(gomoku_tests
  tests/test_board.cpp
  tests/test_windetector.cpp
  tests/test_evaluator.cpp
  tests/test_engine.cpp
  tests/test_engine_dispatch.cpp
  tests/test_neural.cpp
  tests/test_mcts.cpp
  tests/test_game_record.cpp
  tests/test_selfplay.cpp
)
target_link_libraries(gomoku_tests PRIVATE gomoku_core Catch2::Catch2WithMain)
target_compile_options(gomoku_tests PRIVATE ${GOMOKU_WARN_FLAGS})

include(CTest)
catch_discover_tests(gomoku_tests)

# --- Headless training-data tools (developer tooling, never installed) --------
add_executable(gomoku-bootstrap src/tools/bootstrap.cpp)
target_link_libraries(gomoku-bootstrap PRIVATE gomoku_core)
target_compile_options(gomoku-bootstrap PRIVATE ${GOMOKU_WARN_FLAGS})

add_executable(gomoku-selfplay src/tools/self_play.cpp)
target_link_libraries(gomoku-selfplay PRIVATE gomoku_core)
target_compile_options(gomoku-selfplay PRIVATE ${GOMOKU_WARN_FLAGS})

add_executable(gomoku-dump-weights src/tools/dump_weights.cpp)
target_link_libraries(gomoku-dump-weights PRIVATE gomoku_core)
target_compile_options(gomoku-dump-weights PRIVATE ${GOMOKU_WARN_FLAGS})

add_executable(gomoku-nn-eval src/tools/nn_eval.cpp)
target_link_libraries(gomoku-nn-eval PRIVATE gomoku_core)
target_compile_options(gomoku-nn-eval PRIVATE ${GOMOKU_WARN_FLAGS})

# --- GUI application (Qt 6 Widgets) --------------------------------------------
find_package(Qt6 6.4 REQUIRED COMPONENTS Widgets Test)
qt_standard_project_setup()

add_executable(gomoku
  src/main.cpp
  src/ui/MainWindow.hpp
  src/ui/MainWindow.cpp
  src/ui/BoardWidget.hpp
  src/ui/BoardWidget.cpp
  src/ui/GameController.hpp
  src/ui/GameController.cpp
)
target_link_libraries(gomoku PRIVATE gomoku_core Qt6::Widgets)
target_compile_options(gomoku PRIVATE ${GOMOKU_WARN_FLAGS})

# --- Controller state-machine regression test (Qt Core + Qt Test) -------------
add_executable(gomoku_controller_tests
  tests/test_controller.cpp
  src/ui/GameController.hpp
  src/ui/GameController.cpp
)
target_link_libraries(gomoku_controller_tests PRIVATE gomoku_core Qt6::Test)
target_compile_options(gomoku_controller_tests PRIVATE ${GOMOKU_WARN_FLAGS})
add_test(NAME controller COMMAND gomoku_controller_tests)

include(GNUInstallDirs)
install(TARGETS gomoku RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
