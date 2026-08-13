{
  description = "Offline Gomoku — player vs AI (C++23, Qt 6)";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/e7a3ca8092b61ff85b6a45bf863ea2b2d6a661b3";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        qt = pkgs.qt6;
      in
      {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            cmake
            ninja
            gcc14
            gdb
            clang-tools
            qt.qtbase
            qt.wrapQtAppsHook
            catch2_3
          ];
          # Deterministic environment for Qt6 + CMake discovery and for
          # launching the GUI from the build tree.
          shellHook = ''
            export QT_PLUGIN_PATH="${qt.qtbase}/${qt.qtbase.qtPluginPrefix}"
            export CMAKE_PREFIX_PATH="${qt.qtbase.dev}:${pkgs.catch2_3}:$CMAKE_PREFIX_PATH"
          '';
        };

        packages.default = pkgs.stdenv.mkDerivation {
          pname = "gomoku";
          version = "0.1.0";
          src = pkgs.lib.cleanSourceWith {
            src = ./.;
            filter = path: type:
              let base = baseNameOf path;
              in !(base == "build" || base == ".pi" || base == ".direnv" || base == "result" || base == ".git");
          };
          nativeBuildInputs = with pkgs; [ cmake ninja qt.wrapQtAppsHook ];
          buildInputs = with pkgs; [ qt.qtbase catch2_3 ];
          doCheck = true;
          meta = {
            description = "Offline Gomoku game: player vs AI";
            mainProgram = "gomoku";
          };
        };

        apps.default = {
          type = "app";
          program = "${self.packages.${system}.default}/bin/gomoku";
          meta = { mainProgram = "gomoku"; description = "Offline Gomoku game: player vs AI"; };
        };

        checks = {
          default = self.packages.${system}.default;
        };
      });
}
