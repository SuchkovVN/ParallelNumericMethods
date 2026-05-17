#include "common/numeric_utils.hpp"

namespace numeric {
size_t divUp(size_t a, size_t b) {
    return (a + (b - 1)) / b;
}
}  // namespace numeric