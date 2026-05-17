#include "common/memory_utils.hpp"

namespace memory {
void* aligned_alloc(size_t size, size_t alignment) {
#ifdef __MSVC
    return _aligned_malloc(size, alignment);
#else
    return std::aligned_alloc(alignment, size);
#endif
}

void aligned_free(void* mem) {
#ifdef __MSVC
    return _aligned_free(mem);
#else
    return std::free(mem);
#endif
}
}  // namespace memory