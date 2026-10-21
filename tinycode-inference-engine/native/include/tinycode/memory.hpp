// Memory helpers of the native runtime.
#ifndef TINYCODE_MEMORY_HPP
#define TINYCODE_MEMORY_HPP

#include <cstddef>
#include <cstdint>

namespace tinycode {

/// The alignment that the tensor data uses (32 bytes, like GGML).
inline constexpr std::size_t kAlignment = 32;

/// Rounds ``value`` up to the next multiple of ``alignment``.
std::size_t align_up(std::size_t value, std::size_t alignment = kAlignment);

/// Allocates aligned memory; returns ``nullptr`` on failure.
void* aligned_alloc(std::size_t size, std::size_t alignment = kAlignment);

/// Releases memory that was allocated with ``aligned_alloc``.
void aligned_free(void* pointer);

}  // namespace tinycode

#endif  // TINYCODE_MEMORY_HPP
