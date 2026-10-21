// CUDA backend of the native runtime.
//
// The CUDA path is optional: it is only compiled when the CUDA toolkit
// is available.  The Python layer never claims CUDA support unless the
// runtime reports a usable device.
#include "tinycode/backend.hpp"

#if defined(TINYCODE_WITH_CUDA)

#include <cuda_runtime.h>

namespace tinycode {
namespace cuda {

int device_count() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess) {
    return 0;
  }
  return count;
}

}  // namespace cuda
}  // namespace tinycode

extern "C" int32_t tinycode_cuda_device_count(void) { return tinycode::cuda::device_count(); }

#endif  // TINYCODE_WITH_CUDA
