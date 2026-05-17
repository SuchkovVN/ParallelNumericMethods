#pragma once

#include <chrono>
#include <cstdlib>
#include <type_traits>
#include <vector>

namespace testing {

class TestResults {
public:
    struct Result {
        size_t refTime;
        size_t testTime;
        double speedUp;
    };

    TestResults(size_t numTests) : _numTests(numTests), _results(numTests) {}
    ~TestResults() = default;

    void set(size_t pos, size_t refTime, size_t testTime);

    void finalize();

    const Result& getMedian() const;

    const Result& getMin() const;

    const Result& getMax() const;

private:
    size_t _numTests;
    bool _finalized = false;
    std::vector<Result> _results;
};

double convertUsToMs(size_t us);

template <typename Ty>
int64_t compareData(const Ty* ref, const Ty* test, int64_t numElems, double eps = 1e-6) {
    if constexpr (std::is_integral_v<Ty>) {
        for (int64_t i = 0; i < numElems; ++i) {
            if (ref[i] != test[i]) {
                return i;
            }
        }
    } else {
        for (int64_t i = 0; i < numElems; ++i) {
            if (std::abs(ref[i] - test[i]) >= eps) {
                return i;
            }
        }
    }
    return -1;
}

template<typename Fn, typename... Args>
size_t meassureCall(const Fn& f, Args&&... args) {
    auto b = std::chrono::steady_clock::now();
    f(std::forward<Args>(args)...);
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - b).count();
}

}  // namespace testing
