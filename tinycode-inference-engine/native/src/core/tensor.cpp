// Core tensor helpers of the native runtime.
#include "tinycode/tensor.hpp"

namespace tinycode {

std::size_t element_size(DataType dtype) {
  switch (dtype) {
    case DataType::F32:
      return 4;
    case DataType::F16:
      return 2;
    case DataType::I8:
      return 1;
    case DataType::I32:
      return 4;
    default:
      return 0;
  }
}

std::size_t TensorDesc::bytes() const { return elements() * element_size(dtype); }

}  // namespace tinycode
