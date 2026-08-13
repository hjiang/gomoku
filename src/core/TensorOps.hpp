#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

namespace gomoku {

// Low-level tensor operations for the neural network forward pass. All pure,
// Qt-free, channels-first float32. Exposed in a header so the hermetic tests
// can check each primitive against hand-computed values.
//
// Conv semantics match PyTorch: shape (outC, inC, kh, kw), stride 1, no
// dilation, `pad` on every side, zero padding.

// out[o][y][x] = bias[o] + sum_{i,dy,dx} in[i][y+dy-pad][x+dx-pad] * W[o][i][dy][dx]
// (terms with an out-of-range input index are skipped). Spatial size is
// preserved when pad == (kh-1)/2. Output length = outC * inH * inW.
[[nodiscard]] inline std::vector<float> conv2d(std::span<const float> input, int inC, int inH,
                                               int inW, std::span<const float> weight,
                                               std::span<const float> bias, int outC, int kh,
                                               int kw, int pad) {
  assert(inC > 0 && inH > 0 && inW > 0 && outC > 0 && kh > 0 && kw > 0);
  assert(input.size() == static_cast<std::size_t>(inC) * inH * inW);
  assert(weight.size() == static_cast<std::size_t>(outC) * inC * kh * kw);
  assert(bias.size() == static_cast<std::size_t>(outC));

  std::vector<float> out(static_cast<std::size_t>(outC) * inH * inW, 0.0f);
  for (int o = 0; o < outC; ++o) {
    for (int y = 0; y < inH; ++y) {
      for (int x = 0; x < inW; ++x) {
        float acc = bias[o];
        for (int i = 0; i < inC; ++i) {
          for (int dy = 0; dy < kh; ++dy) {
            const int iy = y + dy - pad;
            if (iy < 0 || iy >= inH) {
              continue;
            }
            for (int dx = 0; dx < kw; ++dx) {
              const int ix = x + dx - pad;
              if (ix < 0 || ix >= inW) {
                continue;
              }
              const float xv = input[static_cast<std::size_t>(i) * inH * inW +
                                     static_cast<std::size_t>(iy) * inW + ix];
              const float wv = weight[static_cast<std::size_t>(o) * inC * kh * kw +
                                      static_cast<std::size_t>(i) * kh * kw +
                                      static_cast<std::size_t>(dy) * kw + dx];
              acc += xv * wv;
            }
          }
        }
        out[static_cast<std::size_t>(o) * inH * inW + static_cast<std::size_t>(y) * inW + x] =
            acc;
      }
    }
  }
  return out;
}

// BatchNorm in eval mode, per channel c:
//   y = gamma[c] * (x - mean[c]) / sqrt(var[c] + eps) + beta[c]
// Input shape (C, H, W), gamma/beta/mean/var each length C.
[[nodiscard]] inline std::vector<float> batchNorm(std::span<const float> input, int C, int H,
                                                  int W, std::span<const float> gamma,
                                                  std::span<const float> beta,
                                                  std::span<const float> mean,
                                                  std::span<const float> var, float eps = 1e-5f) {
  assert(C > 0 && H > 0 && W > 0);
  assert(input.size() == static_cast<std::size_t>(C) * H * W);
  assert(gamma.size() == static_cast<std::size_t>(C));
  assert(beta.size() == static_cast<std::size_t>(C));
  assert(mean.size() == static_cast<std::size_t>(C));
  assert(var.size() == static_cast<std::size_t>(C));

  const int plane = H * W;
  std::vector<float> out(input.size());
  for (int c = 0; c < C; ++c) {
    const float invStd = 1.0f / std::sqrt(var[c] + eps);
    for (int i = 0; i < plane; ++i) {
      const float x = input[static_cast<std::size_t>(c) * plane + i];
      out[static_cast<std::size_t>(c) * plane + i] = gamma[c] * (x - mean[c]) * invStd + beta[c];
    }
  }
  return out;
}

// Fully-connected layer: out[j] = bias[j] + sum_i W[j][i] * in[i]; W is (out, in).
[[nodiscard]] inline std::vector<float> linear(std::span<const float> input,
                                               std::span<const float> weight,
                                               std::span<const float> bias, int out, int in) {
  assert(out > 0 && in > 0);
  assert(input.size() == static_cast<std::size_t>(in));
  assert(weight.size() == static_cast<std::size_t>(out) * in);
  assert(bias.size() == static_cast<std::size_t>(out));

  std::vector<float> result(static_cast<std::size_t>(out));
  for (int j = 0; j < out; ++j) {
    float acc = bias[j];
    for (int i = 0; i < in; ++i) {
      acc += weight[static_cast<std::size_t>(j) * in + i] * input[i];
    }
    result[j] = acc;
  }
  return result;
}

// ReLU in place on the caller's vector (move for reuse).
[[nodiscard]] inline std::vector<float> relu(std::vector<float> input) {
  for (float& v : input) {
    v = std::max(0.0f, v);
  }
  return input;
}

// Softmax over the whole vector. All values must be finite; callers that need
// to mask illegal actions should set those logits to -inf first.
[[nodiscard]] inline std::vector<float> softmax(std::span<const float> logits) {
  assert(!logits.empty());
  float maxLogit = logits[0];
  for (const float v : logits) {
    maxLogit = std::max(maxLogit, v);
  }
  std::vector<float> p(logits.size());
  float sum = 0.0f;
  for (std::size_t i = 0; i < logits.size(); ++i) {
    p[i] = std::exp(logits[i] - maxLogit);
    sum += p[i];
  }
  for (float& v : p) {
    v /= sum;
  }
  return p;
}

}  // namespace gomoku
