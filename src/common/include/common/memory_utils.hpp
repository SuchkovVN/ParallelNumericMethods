#pragma once

#include <cstdlib>

namespace memory {
void* aligned_alloc(size_t size, size_t alignment);
void aligned_free(void* mem);
}  // namespace memory
