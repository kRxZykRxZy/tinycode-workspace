// Tests of the native runtime.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "tinycode/operators.hpp"

namespace {

int failures = 0;

void check(bool condition, const char* message) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    ++failures;
  }
}

void test_version() { check(tinycode_version() >= 1, "the version must be at least 1"); }

void test_matmul() {
  const int32_t m = 2;
  const int32_t n = 3;
  const int32_t k = 2;
  std::vector<float> a = {1, 2, 3, 4};
  std::vector<float> b = {1, 0, 0, 1, 1, 1};
  std::vector<float> c(m * n, 0.0f);
  tinycode_matmul_f32(a.data(), b.data(), c.data(), m, n, k);
  check(std::fabs(c[0] - 1.0f) < 1e-5f, "matmul[0][0]");
  check(std::fabs(c[1] - 2.0f) < 1e-5f, "matmul[0][1]");
  check(std::fabs(c[3] - 4.0f) < 1e-5f, "matmul[1][0]");
}

void test_rmsnorm() {
  std::vector<float> x = {3.0f, 4.0f};
  std::vector<float> gain = {1.0f, 1.0f};
  tinycode_rmsnorm_f32(x.data(), gain.data(), 2, 1e-5f);
  const float expected = 1.0f / std::sqrt(12.5f);
  check(std::fabs(x[0] - 3.0f * expected) < 1e-5f, "rmsnorm[0]");
  check(std::fabs(x[1] - 4.0f * expected) < 1e-5f, "rmsnorm[1]");
}

void test_softmax() {
  std::vector<float> x = {0.0f, 0.0f};
  tinycode_softmax_f32(x.data(), 2);
  check(std::fabs(x[0] - 0.5f) < 1e-5f, "softmax[0]");
  check(std::fabs(x[0] + x[1] - 1.0f) < 1e-5f, "softmax sums to one");
}

void test_silu() {
  std::vector<float> x = {0.0f};
  tinycode_silu_f32(x.data(), 1);
  check(std::fabs(x[0]) < 1e-6f, "silu(0) is zero");
}

void test_add() {
  std::vector<float> a = {1.0f, 2.0f};
  std::vector<float> b = {3.0f, 4.0f};
  tinycode_add_f32(a.data(), b.data(), 2);
  check(std::fabs(a[0] - 4.0f) < 1e-6f, "add[0]");
  check(std::fabs(a[1] - 6.0f) < 1e-6f, "add[1]");
}

}  // namespace

int main() {
  test_version();
  test_matmul();
  test_rmsnorm();
  test_softmax();
  test_silu();
  test_add();
  if (failures == 0) {
    std::printf("all native tests passed\n");
    return 0;
  }
  std::printf("%d native test(s) failed\n", failures);
  return 1;
}
