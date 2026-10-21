// Public C interface of the TinyCode native runtime.
#ifndef TINYCODE_OPERATORS_HPP
#define TINYCODE_OPERATORS_HPP

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

/// Returns the version of the native library.
int32_t tinycode_version(void);

/// Returns 1 when the AVX2 kernels are compiled in, 0 otherwise.
int32_t tinycode_has_avx2(void);

/// Returns 1 when the NEON kernels are compiled in, 0 otherwise.
int32_t tinycode_has_neon(void);

/// C = A (M x K) * B (K x N); all matrices are row major float32.
void tinycode_matmul_f32(const float* A, const float* B, float* C, int32_t M, int32_t N,
                         int32_t K);

/// RMS normalisation of a float32 buffer: x = x / sqrt(mean(x^2) + eps) * gain.
void tinycode_rmsnorm_f32(float* x, const float* gain, int32_t n, float eps);

/// In-place softmax of a float32 buffer.
void tinycode_softmax_f32(float* x, int32_t n);

/// In-place SiLU activation of a float32 buffer.
void tinycode_silu_f32(float* x, int32_t n);

/// In-place addition: a += b.
void tinycode_add_f32(float* a, const float* b, int32_t n);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TINYCODE_OPERATORS_HPP
