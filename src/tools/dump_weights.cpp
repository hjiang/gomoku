// Headless .gnn introspection tool: parses a weight file with Weights::parse
// and dumps the header plus every tensor in the exact file order. Used by the
// Stage 3 round-trip gate (training/tests/test_gnn_roundtrip.py) to prove the
// Python-written .gnn bytes are read back by the C++ loader identically.
// Developer tooling only — never installed, never part of the shipped game.

#include "core/Weights.hpp"

#include <cstdint>
#include <cstdlib>
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

// Emits '<name> <n_floats>' then the tensor's floats, one per line. The name
// must exactly match the gnn_format.py tensor names so the round-trip gate can
// cross-check Python-vs-C++ layout non-circularly.
void dump(const char* name, const std::vector<float>& v) {
  std::cout << name << ' ' << v.size() << '\n';
  for (const float x : v) {
    std::cout << x << '\n';
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::string model;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--model" && i + 1 < argc) {
      model = argv[++i];
    } else {
      std::cerr << "usage: gomoku-dump-weights --model FILE.gnn\n";
      return 1;
    }
  }
  if (model.empty()) {
    std::cerr << "usage: gomoku-dump-weights --model FILE.gnn\n";
    return 1;
  }

  std::vector<std::uint8_t> bytes;
  if (!readFile(model, bytes)) {
    std::cerr << "cannot read model file " << model << '\n';
    return 1;
  }
  const std::optional<Weights> w = Weights::parse(bytes);
  if (!w) {
    std::cerr << "failed to parse " << model << '\n';
    return 1;
  }

  // 9 significant digits round-trip float32 exactly (%.9g).
  std::cout << std::setprecision(9);
  std::cout << "num_blocks " << w->numBlocks << '\n';
  std::cout << "channels " << w->channels << '\n';

  // Tensor order must mirror Weights::parse exactly.
  dump("inputConvW", w->inputConvW);
  dump("inputConvB", w->inputConvB);
  dump("inputBnGamma", w->inputBnGamma);
  dump("inputBnBeta", w->inputBnBeta);
  dump("inputBnMean", w->inputBnMean);
  dump("inputBnVar", w->inputBnVar);
  for (const Weights::Block& b : w->blocks) {
    dump("conv1W", b.conv1W);
    dump("conv1B", b.conv1B);
    dump("bn1Gamma", b.bn1Gamma);
    dump("bn1Beta", b.bn1Beta);
    dump("bn1Mean", b.bn1Mean);
    dump("bn1Var", b.bn1Var);
    dump("conv2W", b.conv2W);
    dump("conv2B", b.conv2B);
    dump("bn2Gamma", b.bn2Gamma);
    dump("bn2Beta", b.bn2Beta);
    dump("bn2Mean", b.bn2Mean);
    dump("bn2Var", b.bn2Var);
  }
  dump("policyConvW", w->policyConvW);
  dump("policyConvB", w->policyConvB);
  dump("policyFcW", w->policyFcW);
  dump("policyFcB", w->policyFcB);
  dump("valueConvW", w->valueConvW);
  dump("valueConvB", w->valueConvB);
  dump("valueFc1W", w->valueFc1W);
  dump("valueFc1B", w->valueFc1B);
  dump("valueFc2W", w->valueFc2W);
  dump("valueFc2B", w->valueFc2B);
  return 0;
}
