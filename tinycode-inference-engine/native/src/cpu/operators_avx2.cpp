// AVX2 accelerated kernels.
//
// The file is only compiled when the compiler supports -mavx2; the
// dispatch in the Python layer checks the runtime capability.
#include <cmath>
#include <cstdint>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)

#include <immintrin.h>

namespace tinycode {
namespace avx2 {

void matmul_f32(const float* a, const float* b, float* c, int32_t m, int32_t n, int32_t k) {
  const __m256 zero = _mm256_setzero_ps();
  for (int32_t row = 0; row < m; ++row) {
    for (int32_t col = 0; col < n; col += 8) {
      __m256 acc = zero;
      const int32_t block = (n - col) < 8 ? (n - col) : 8;
      for (int32_t inner = 0; inner < k; ++inner) {
        const __m256 value = _mm256_set1_ps(a[row * k + inner]);
        if (block == 8) {
          const __m256 weights = _mm256_loadu_ps(b + inner * n + col);
          acc = _mm256_fmadd_ps(value, weights, acc);
        } else {
          float partial[8] = {0, 0, 0, 0, 0, 0, 0, 0};
          _mm256_storeu_ps(partial, acc);
          for (int32_t lane = 0; lane < block; ++lane) {
            partial[lane] += a[row * k + inner] * b[inner * n + col + lane];
          }
          acc = _mm256_loadu_ps(partial);
        }
      }
      _mm256_storeu_ps(c + row * n + col, acc);
    }
  }
}

void rmsnorm_f32(float* x, const float* gain, int32_t n, float eps) {
  if (n <= 0) {
    return;
  }
  __m256 acc = _mm256_setzero_ps();
  int32_t i = 0;
  for (; i + 8 <= n; i += 8) {
    const __m256 values = _mm256_loadu_ps(x + i);
    acc = _mm256_add_ps(acc, _mm256_mul_ps(values, values));
  }
  float buffer[8];
  _mm256_storeu_ps(buffer, acc);
  float sum = 0.0f;
  for (int32_t lane = 0; lane < 8; ++lane) {
    sum += buffer[lane];
  }
  for (; i < n; ++i) {
    sum += x[i] * x[i];
  }
  const float inv = 1.0f / std::sqrt(sum / static_cast<float>(n) + eps);
  i = 0;
  for (; i + 8 <= n; i += 8) {
    const __m256 values = _mm256_loadu_ps(x + i);
    const __m256 gains = _mm256_loadu_ps(gain + i);
    _mm256_storeu_ps(x + i, _mm256_mul_ps(_mm256_mul_ps(values, _mm256_set1_ps(inv)), gains));
  }
  for (; i < n; ++i) {
    x[i] = x[i] * inv * gain[i];
  }
}

void silu_f32(float* x, int32_t n) {
  int32_t i = 0;
  for (; i + 8 <= n; i += 8) {
    const __m256 values = _mm256_loadu_ps(x + i);
    const __m256 neg = _mm256_sub_ps(_mm256_setzero_ps(), values);
    const __m256 exp = _mm256_exp_ps(neg);
    const __m256 denom = _mm256_add_ps(_mm256_set1_ps(1.0f), exp);
    _mm256_storeu_ps(x + i, _mm256_div_ps(values, denom));
  }
  for (; i < n; ++i) {
    x[i] = x[i] / (1.0f + std::exp(-x[i]));
  }
}

}  // namespace avx2
}  // namespace tinycode

extern "C" void tinycode_avx2_matmul_f32(const float* A, const float* B, float* C, int32_t M,
                                         int32_t N, int32_t K) {
  tinycode::avx2::matmul_f32(A, B, C, M, N, K);
}

extern "C" void tinycode_avx2_rmsnorm_f32(float* x, const float* gain, int32_t n, float eps) {
  tinycode::avx2::rmsnorm_f32(x, gain, n, eps);
}

extern "C" void tinycode_avx2_silu_f32(float* x, int32_t n) { tinycode::avx2::silu_f32(x, n); }

#endif  // x86
