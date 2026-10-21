// Tensor description of the native runtime.
#ifndef TINYCODE_TENSOR_HPP
#define TINYCODE_TENSOR_HPP

#include <cstddef>
#include <cstdint>

namespace tinycode {

/// The element types the native runtime understands.
enum class DataType : std::uint8_t {
  F32 = 0,
  F16 = 1,
  I8 = 2,
  I32 = 3,
};

/// A description of a contiguous, row major tensor.
struct TensorDesc {
  DataType dtype = DataType::F32;
  std::size_t rows = 0;
  std::size_t cols = 0;
  std::size_t stride = 0;  // elements between two rows

  std::size_t elements() const { return rows * cols; }
  std::size_t bytes() const;
};

/// The number of bytes of one element of ``dtype``.
std::size_t element_size(DataType dtype);

}  // namespace tinycode

#endif  // TINYCODE_TENSOR_HPP
