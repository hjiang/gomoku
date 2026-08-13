#pragma once

// Shared test helper: builds a valid .gnn byte buffer with all-zero tensors
// for `numBlocks` blocks of `channels` channels. A zero network gives a
// uniform policy and value 0, so MCTS relies purely on terminal win/loss
// detection. The tensor order mirrors Weights::parse exactly (see
// src/core/Weights.cpp and docs/plans/PLAN-neural-mcts.md); keeping it in one
// place means the layout knowledge is not duplicated across test files.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

inline std::vector<std::uint8_t> makeZeroGnn(int numBlocks, int channels) {
  std::vector<std::uint8_t> buf;
  const char* magic = "GOMOKUNET";
  buf.insert(buf.end(), magic, magic + 9);

  const auto putU32 = [&](std::uint32_t v) {
    for (int i = 0; i < 4; ++i) {
      buf.push_back(static_cast<std::uint8_t>((v >> (8 * i)) & 0xFF));
    }
  };
  const auto putF32 = [&](float v) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &v, sizeof(bits));
    putU32(bits);
  };
  const auto zeros = [&](std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) {
      putF32(0.0f);
    }
  };

  putU32(1);  // version
  putU32(static_cast<std::uint32_t>(numBlocks));
  putU32(static_cast<std::uint32_t>(channels));

  const int C = channels;
  zeros(C * 4 * 9);                        // input conv weight (C,4,3,3)
  zeros(C);                                // input conv bias
  zeros(C); zeros(C); zeros(C); zeros(C);  // input bn gamma/beta/mean/var
  for (int b = 0; b < numBlocks; ++b) {
    zeros(C * C * 9); zeros(C);                 // conv1 W, b
    zeros(C); zeros(C); zeros(C); zeros(C);     // bn1 gamma/beta/mean/var
    zeros(C * C * 9); zeros(C);                 // conv2 W, b
    zeros(C); zeros(C); zeros(C); zeros(C);     // bn2 gamma/beta/mean/var
  }
  zeros(2 * C * 9);  // policy conv weight (2,C,3,3)
  zeros(2);          // policy conv bias
  zeros(225 * 450);  // policy fc weight
  zeros(225);        // policy fc bias
  zeros(C * 9);      // value conv weight (1,C,3,3)
  zeros(1);          // value conv bias
  zeros(256 * 225);  // value fc1 weight
  zeros(256);        // value fc1 bias
  zeros(256);        // value fc2 weight (1,256)
  zeros(1);          // value fc2 bias
  return buf;
}
