// ARM NEON accelerated kernels.
//
// The file is only compiled on ARM hosts with NEON support; the Python
// layer verifies the capability at runtime before using them.
#include <cmath>
#include <cstdint>

#if defined(__aarch64__) || defined(__ARM_NEON) || defined(_M_ARM64)

#include <arm_neon.h>

namespace tinycode {
namespace neon {

void rmsnorm_f32(float* x, const float* gain, int32_t n, float eps) {
  if (n <= 0) {
    return;
  }
  float32x4_t acc = vdupq_n_f32(0.0f);
  int32_t i = 0;
  for (; i + 4 <= n; i += 4) {
    const float32x4_t values = vld1q_f32(x + i);
    acc = vmlaq_f32(acc, values, values);
  }
  float buffer[4];
  vst1q_f32(buffer, acc);
  float sum = buffer[0] + buffer[1] + buffer[2] + buffer[3];
  for (; i < n; ++i) {
    sum += x[i] * x[i];
  }
  const float inv = 1.0f / std::sqrt(sum / static_cast<float>(n) + eps);
  i = 0;
  for (; i + 4 <= n; i += 4) {
    const float32x4_t values = vld1q_f32(x + i);
    const float32x4_t gains = vld1q_f32(gain + i);
    vst1q_f32(x + i, vmulq_f32(vmulq_n_f32(values, inv), gains));
  }
  for (; i < n; ++i) {
    x[i] = x[i] * inv * gain[i];
  }
}

void silu_f32(float* x, int32_t n) {
  int32_t i = 0;
  for (; i + 4 <= n; i += 4) {
    const float32x4_t values = vld1q_f32(x + i);
    const float32x4_t neg = vnegq_f32(values);
    const float32x4_t exp = vexpq_f32(neg);
    const float32x4_t denom = vaddq_f32(vdupq_n_f32(1.0f), exp);
    vst1q_f32(x + i, vdivq_f32(values, denom));
  }
  for (; i < n; ++i) {
    x[i] = x[i] / (1.0f + std::exp(-x[i]));
  }
}

}  // namespace neon
}  // namespace tinycode

extern "C" void tinycode_neon_rmsnorm_f32(float* x, const float* gain, int32_t n, float eps) {
  tinycode::neon::rmsnorm_f32(x, gain, n, eps);
}

extern "C" void tinycode_neon_silu_f32(float* x, int32_t n) { tinycode::neon::silu_f32(x, n); }

#endif  // ARM
