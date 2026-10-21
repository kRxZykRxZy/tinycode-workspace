// Backend description of the native runtime.
#ifndef TINYCODE_BACKEND_HPP
#define TINYCODE_BACKEND_HPP

#include <cstdint>
#include <string>

namespace tinycode {

/// The execution providers the native runtime knows about.
enum class BackendKind : std::uint8_t {
  Cpu = 0,
  Cuda = 1,
  IntelGpu = 2,
  ArmNeon = 3,
};

/// The capabilities of a native backend.
struct BackendInfo {
  BackendKind kind = BackendKind::Cpu;
  bool accelerated = false;
  bool supports_avx2 = false;
  bool supports_neon = false;
  std::string provider = "scalar";
};

/// Returns the information about the backend that was compiled in.
BackendInfo backend_info();

}  // namespace tinycode

#endif  // TINYCODE_BACKEND_HPP
