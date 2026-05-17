#include "common/test_utils.hpp"

#include <stdexcept>
#include <algorithm>

namespace testing {
void TestResults::set(size_t pos, size_t refTime, size_t testTime) {
    _results[pos].refTime = refTime;
    _results[pos].testTime = testTime;
    _results[pos].speedUp = (1.l - static_cast<double>(testTime) / refTime) * 100.l;
}

void TestResults::finalize() {
    std::sort(_results.begin(), _results.end(), [](const auto& x, const auto& y) -> bool {
        return x.speedUp < y.speedUp;
    });
    _finalized = true;
}

const TestResults::Result& TestResults::getMedian() const {
    if (!_finalized) {
        throw std::runtime_error("Test results wasn't finalized yet");
    }

    size_t mIdx = _numTests % 2 ? (_numTests - 1) / 2 : _numTests / 2;
    return _results[mIdx];
}

const TestResults::Result& TestResults::getMin() const {
    if (!_finalized) {
        throw std::runtime_error("Test results wasn't finalized yet");
    }

    return _results[0];
}

const TestResults::Result& TestResults::getMax() const {
    if (!_finalized) {
        throw std::runtime_error("Test results wasn't finalized yet");
    }

    return _results[_numTests - 1];
}

double convertUsToMs(size_t us) {
    return static_cast<double>(us) / 1'000'000;
}
}  // namespace testing