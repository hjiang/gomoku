// Headless forward-pass tool: loads a .gnn model, reads 900 little-endian f32
// input planes (the exact BoardEncoder::encode output), runs NeuralNet::evaluate
// and prints the 225 policy logits then the value, one per line. Used by the
// Stage 3 C++ <-> PyTorch forward gate (training/tests/test_forward.py).
// Developer tooling only — never installed, never part of the shipped game.

#include "core/BoardEncoder.hpp"
#include "core/NeuralNet.hpp"
#include "core/Weights.hpp"
#include "core/types.hpp"

#include <bit>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

using namespace gomoku;

namespace {

bool readFile(const std::string& path, std::vector<std::uint8_t>& bytes) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return false;
  }
  in.seekg(0, std::ios::end);
  const std::streamoff size = in.tellg();
  if (size < 0) {
    return false;
  }
  in.seekg(0, std::ios::beg);
  bytes.resize(static_cast<std::size_t>(size));
  if (size > 0) {
    in.read(reinterpret_cast<char*>(bytes.data()), size);
  }
  return static_cast<bool>(in);
}

bool readPlanes(const std::string& path, std::vector<float>& planes) {
  std::vector<std::uint8_t> bytes;
  if (!readFile(path, bytes)) {
    return false;
  }
  constexpr std::size_t kPlaneBytes = BoardEncoder::kInputSize * sizeof(float);
  if (bytes.size() != kPlaneBytes) {
    std::cerr << "expected " << kPlaneBytes << " bytes (900 little-endian f32 planes), got "
              << bytes.size() << '\n';
    return false;
  }
  static_assert(std::endian::native == std::endian::little,
                "planes loader reads little-endian floats");
  planes.resize(BoardEncoder::kInputSize);
  std::memcpy(planes.data(), bytes.data(), kPlaneBytes);  // little-endian host
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  std::string model;
  std::string input;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if ((arg == "--model" || arg == "--input") && i + 1 < argc) {
      const std::string value = argv[++i];
      if (arg == "--model") {
        model = value;
      } else {
        input = value;
      }
    } else {
      std::cerr << "usage: gomoku-nn-eval --model FILE.gnn --input FILE.planes\n";
      return 1;
    }
  }
  if (model.empty() || input.empty()) {
    std::cerr << "usage: gomoku-nn-eval --model FILE.gnn --input FILE.planes\n";
    return 1;
  }

  std::vector<std::uint8_t> modelBytes;
  if (!readFile(model, modelBytes)) {
    std::cerr << "cannot read model file " << model << '\n';
    return 1;
  }
  const std::optional<Weights> w = Weights::parse(modelBytes);
  if (!w) {
    std::cerr << "failed to parse " << model << '\n';
    return 1;
  }

  std::vector<float> planes;
  if (!readPlanes(input, planes)) {
    std::cerr << "cannot read input planes " << input << '\n';
    return 1;
  }

  const NeuralNet net(*w);
  const NeuralNet::Output out = net.evaluate(planes);

  std::cout << std::setprecision(9);
  for (const float logit : out.policyLogits) {
    std::cout << logit << '\n';
  }
  std::cout << out.value << '\n';
  return 0;
}
