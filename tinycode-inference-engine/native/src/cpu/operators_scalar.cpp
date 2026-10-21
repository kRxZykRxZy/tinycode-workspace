// Scalar implementations of the TinyCode operators.
#include "tinycode/operators.hpp"

#include <cmath>
#include <cstring>

namespace tinycode {
namespace scalar {

void matmul_f32(const float* a, const float* b, float* c, int32_t m, int32_t n, int32_t k) {
  for (int32_t row = 0; row < m; ++row) {
    for (int32_t col = 0; col < n; ++col) {
      float sum = 0.0f;
      for (int32_t inner = 0; inner < k; ++inner) {
        sum += a[row * k + inner] * b[inner * n + col];
      }
      c[row * n + col] = sum;
    }
  }
}

void rmsnorm_f32(float* x, const float* gain, int32_t n, float eps) {
  if (n <= 0) {
    return;
  }
  double sum = 0.0;
  for (int32_t i = 0; i < n; ++i) {
    sum += static_cast<double>(x[i]) * static_cast<double>(x[i]);
  }
  const float inv = 1.0f / std::sqrt(static_cast<float>(sum / n) + eps);
  for (int32_t i = 0; i < n; ++i) {
    x[i] = x[i] * inv * gain[i];
  }
}

void softmax_f32(float* x, int32_t n) {
  if (n <= 0) {
    return;
  }
  float maximum = x[0];
  for (int32_t i = 1; i < n; ++i) {
    if (x[i] > maximum) {
      maximum = x[i];
    }
  }
  float total = 0.0f;
  for (int32_t i = 0; i < n; ++i) {
    x[i] = std::exp(x[i] - maximum);
    total += x[i];
  }
  const float inv = total > 0.0f ? 1.0f / total : 0.0f;
  for (int32_t i = 0; i < n; ++i) {
    x[i] *= inv;
  }
}

void silu_f32(float* x, int32_t n) {
  for (int32_t i = 0; i < n; ++i) {
    x[i] = x[i] / (1.0f + std::exp(-x[i]));
  }
}

void add_f32(float* a, const float* b, int32_t n) {
  for (int32_t i = 0; i < n; ++i) {
    a[i] += b[i];
  }
}

}  // namespace scalar
}  // namespace tinycode

// ---------------------------------------------------------------------------
// C interface
// ---------------------------------------------------------------------------
extern "C" {

int32_t tinycode_version(void) { return 1; }

int32_t tinycode_has_avx2(void) {
#if defined(TINYCODE_USE_AVX2)
  return 1;
#else
  return 0;
#endif
}

int32_t tinycode_has_neon(void) {
#if defined(TINYCODE_USE_NEON)
  return 1;
#else
  return 0;
#endif
}

void tinycode_matmul_f32(const float* A, const float* B, float* C, int32_t M, int32_t N,
                         int32_t K) {
  tinycode::scalar::matmul_f32(A, B, C, M, N, K);
}

void tinycode_rmsnorm_f32(float* x, const float* gain, int32_t n, float eps) {
  tinycode::scalar::rmsnorm_f32(x, gain, n, eps);
}

void tinycode_softmax_f32(float* x, int32_t n) { tinycode::scalar::softmax_f32(x, n); }

void tinycode_silu_f32(float* x, int32_t n) { tinycode::scalar::silu_f32(x, n); }

void tinycode_add_f32(float* a, const float* b, int32_t n) { tinycode::scalar::add_f32(a, b, n); }

}  // extern "C"
