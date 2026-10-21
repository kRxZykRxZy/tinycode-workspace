// Memory helpers of the native runtime.
#include "tinycode/memory.hpp"

#include <cstdlib>
#include <new>

namespace tinycode {

std::size_t align_up(std::size_t value, std::size_t alignment) {
  if (alignment == 0) {
    return value;
  }
  const std::size_t remainder = value % alignment;
  return remainder == 0 ? value : value + (alignment - remainder);
}

void* aligned_alloc(std::size_t size, std::size_t alignment) {
  if (size == 0) {
    size = 1;
  }
#if defined(_MSC_VER)
  return _aligned_malloc(size, alignment);
#else
  void* pointer = nullptr;
  if (posix_memalign(&pointer, alignment, size) != 0) {
    return nullptr;
  }
  return pointer;
#endif
}

void aligned_free(void* pointer) {
  if (pointer == nullptr) {
    return;
  }
#if defined(_MSC_VER)
  _aligned_free(pointer);
#else
  std::free(pointer);
#endif
}

}  // namespace tinycode
