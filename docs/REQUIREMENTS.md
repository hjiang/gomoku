# Gomoku — Requirements

## Purpose
An offline, single-player **Gomoku** (five-in-a-row) game: the human plays against
a computer opponent on a local board. No network access, no accounts, no server.

## Scope
- One game mode: **Player vs AI** (human vs computer). No PvP, no online play.
- **Freestyle gomoku** rules (not Renju): five *or more* consecutive stones of one
  color in a row, column, or diagonal wins. No forbidden moves for either side.
- Standard **15×15** board as the default; board size is a compile-time constant so
  it could be raised later (e.g. 19×19) with minimal change.

## Functional requirements

| ID   | Requirement |
|------|-------------|
| FR1  | Render a 15×15 grid board with stones (black/white) drawn at intersections. |
| FR2  | Player controls **Black** and moves first; AI plays **White** (configurable). |
| FR3  | Turns alternate strictly; clicking an occupied intersection is ignored. |
| FR4  | Clicking a legal intersection places the player's stone immediately. |
| FR5  | After each move, the game checks for a win (5+ in a row, any of 4 directions). |
| FR6  | A win is announced and the winning line is highlighted; input is frozen. |
| FR7  | A full board with no winner is announced as a **draw**. |
| FR8  | AI difficulty levels (Easy / Medium / Hard) map to search depth and/or a time budget. |
| FR9  | **New game**, **Undo**, and (optionally) **Resign** controls. |
| FR10 | The last move is visually marked; a status bar shows whose turn it is. |
| FR11 | The UI stays responsive while the AI is thinking (AI runs off the UI thread). |
| FR12 | The game runs fully offline with no network calls of any kind. |

## Non-functional requirements

| ID   | Requirement |
|------|-------------|
| NFR1 | **Modern C++**: C++23, RAII, no raw `new`/`delete`, `constexpr` where practical, no global mutable state. |
| NFR2 | **Qt 6** Widgets (not QML) for the UI. |
| NFR3 | **Reproducible toolchain via Nix**: a `flake.nix` provides the compiler, CMake, Qt 6, and the test framework. `nix develop` gives a ready dev shell; `nix build`/`nix run` produce and run the game. |
| NFR4 | **Testability**: all game rules and the AI live in a Qt-free core library with unit tests. |
| NFR5 | **Responsiveness**: hard-difficulty AI returns a move within a bounded time (target ≤ 2 s). |
| NFR6 | No compiler warnings under `-Wall -Wextra` (and `-Werror` in CI checks). |

## Constraints & assumptions
- Freestyle gomoku: an "overline" (6+ in a row) **counts as a win**.
- Single human player; the AI is a deterministic search (no learned model, no network).
- Target platform: Linux (NixOS/nix on any Linux); the code itself is portable C++.

## Out of scope (non-goals)
- Renju tournament rules / forbidden moves.
- Online play, multiplayer, leaderboards, or persistence of game history.
- Sounds, animations beyond the minimum, theming engine.
- Reinforcement-learning / neural-network AI.
- Mobile or web deployment.

## Acceptance criteria
1. `nix develop` drops into a shell where `cmake --build build` succeeds.
2. `ctest --test-dir build` passes all unit tests with no warnings.
3. `nix run .` launches the game window and a full game can be played to a win/draw.
4. On the hardest difficulty, the AI blocks an obvious open four and completes its own win.
5. The UI never freezes during the AI's turn.
