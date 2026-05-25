#include "common/memory_utils.hpp"
#include "common/numeric_utils.hpp"
#include "common/test_utils.hpp"

#include <chrono>
#include <cstring>
#include <immintrin.h>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <random>
#include <stdlib.h>
#include <string>
#include <vector>
#include <algorithm>

static constexpr int64_t maskarr[] = { -1, -1, -1, -1, 0, 0, 0, 0 };
static constexpr double cvNanosecSec = 1'000'000.l;

struct SubMatrix {
    double* ptr;
    int n;
    int m;
    int stride;
};

class MatBuffer {
public:
    explicit MatBuffer(size_t size, size_t alignment = 32) : _size(numeric::divUp(size * sizeof(double), alignment) * alignment), _align(alignment) {
        _buffer = memory::aligned_alloc(_size, _align);
    }
    ~MatBuffer() {
        memory::aligned_free(_buffer);
    }

    MatBuffer(const MatBuffer&) = delete;
    MatBuffer& operator=(const MatBuffer&) = delete;

    double* data() {
        return static_cast<double*>(_buffer);
    }

    void memset(double val) {
        ::memset(_buffer, val, _size);
    }

private:
    void* _buffer;
    size_t _align;
    size_t _size;
};

//
// Main funcs
//

// Solve AXT = BT for X
inline void solveLinearSystemTransposed(const SubMatrix& A, const SubMatrix& B, SubMatrix& X) {
    const int packSize = 4;
#pragma omp parallel for
    for (int k = 0; k < B.n; ++k) {
        for (int j = 0; j < A.m; ++j) {
            double acc = B.ptr[j + B.stride * k];

            int quot = j / packSize;
            int rem = j % packSize;
            int X_start = X.stride * k;
            int A_start = A.stride * j;
            int pos = 0;
            __m256d sum = _mm256_setzero_pd();
            for (int s = 0; s < quot; ++s) {
                pos = s * packSize;
                __m256d A_elems = _mm256_loadu_pd(&A.ptr[pos + A_start]);
                __m256d X_elems = _mm256_loadu_pd(&X.ptr[pos + X_start]);
                sum = _mm256_fmadd_pd(A_elems, X_elems, sum);
            }
            pos = j - rem;

            __m256i mask = _mm256_loadu_si256((const __m256i*)(maskarr + (packSize - rem)));
            __m256d A_tail = _mm256_maskload_pd(&A.ptr[pos + A_start], mask);
            __m256d X_tail = _mm256_maskload_pd(&X.ptr[pos + X_start], mask);
            sum = _mm256_fmadd_pd(A_tail, X_tail, sum);

            sum = _mm256_hadd_pd(sum, sum);
            __m128d hi = _mm256_extractf128_pd(sum, 1);
            __m128d res = _mm_add_sd(_mm256_castpd256_pd128(sum), hi);
            acc -= _mm_cvtsd_f64(res);

            X.ptr[j + X.stride * k] = acc / A.ptr[j + A.stride * j];
        }
    }
}

inline void solveLinearSystemTransposed_NOSIMD(const SubMatrix& A, const SubMatrix& B, SubMatrix& X) {
#pragma omp parallel for
    for (int k = 0; k < B.n; ++k) {
        for (int j = 0; j < A.m; ++j) {
            int X_start = X.stride * k;
            int A_start = A.stride * j;

            double acc = B.ptr[j + B.stride * k];
            for (int s = 0; s < j; ++s) {
                acc += A.ptr[s + A_start] * X.ptr[s + X_start];
            }

            X.ptr[j + X_start] = acc / A.ptr[j + A_start];
        }
    }
}

inline void choleskyDecomp(const SubMatrix& A, SubMatrix& L) {
    constexpr int packSize = 4;
    for (int j = 0; j < A.m; ++j) {
        int L_start = L.stride * j;
        double acc = A.ptr[j + A.stride * j];
        int quot = j / packSize;
        int rem = j % packSize;

        __m256d sum = _mm256_setzero_pd();
        __m256d curr_val = _mm256_setzero_pd();
        for (int s = 0; s < quot; ++s) {
            curr_val = _mm256_loadu_pd(&L.ptr[s * packSize + L_start]);
            sum = _mm256_fmadd_pd(curr_val, curr_val, sum);
        }
        __m256i mask = _mm256_loadu_si256((const __m256i*)(maskarr + (packSize - rem)));
        curr_val = _mm256_maskload_pd(&L.ptr[quot * packSize + L_start], mask);
        sum = _mm256_fmadd_pd(curr_val, curr_val, sum);

        sum = _mm256_hadd_pd(sum, sum);
        __m128d hi = _mm256_extractf128_pd(sum, 1);
        __m128d res = _mm_add_sd(_mm256_castpd256_pd128(sum), hi);
        acc -= _mm_cvtsd_f64(res);

        acc = std::sqrt(acc);
        L.ptr[j + L_start] = acc;

#pragma omp parallel for private(acc, quot, rem, sum) schedule(dynamic)
        for (int s = j + 1; s < L.n; ++s) {
            acc = A.ptr[j + A.stride * s];

            quot = j / packSize;
            rem = j % packSize;
            sum = _mm256_setzero_pd();
            for (int k = 0; k < quot; ++k) {
                __m256d a = _mm256_loadu_pd(&L.ptr[k * packSize + L_start]);
                __m256d b = _mm256_loadu_pd(&L.ptr[k * packSize + L.stride * s]);
                sum = _mm256_fmadd_pd(a, b, sum);
            }
            __m256i mask = _mm256_loadu_si256((const __m256i*)(maskarr + (packSize - rem)));
            __m256d a_tail = _mm256_maskload_pd(&L.ptr[quot * packSize + L_start], mask);
            __m256d b_tail = _mm256_maskload_pd(&L.ptr[quot * packSize + L.stride * s], mask);
            sum = _mm256_fmadd_pd(a_tail, b_tail, sum);

            sum = _mm256_hadd_pd(sum, sum);
            __m128d hi = _mm256_extractf128_pd(sum, 1);
            __m128d res = _mm_add_sd(_mm256_castpd256_pd128(sum), hi);
            acc -= _mm_cvtsd_f64(res);

            L.ptr[j + L.stride * s] = acc / L.ptr[j + L_start];
        }
    }
}

inline void choleskyDecomp_NOSIMD(const SubMatrix& A, SubMatrix& L) {
    for (int j = 0; j < A.m; ++j) {
        int L_start = L.stride * j;
        ;
        double acc = A.ptr[j + A.stride * j];
        for (int s = 0; s < j; ++s) {
            acc += L.ptr[s + L_start] * L.ptr[s + L_start];
        }

        acc = std::sqrt(acc);
        L.ptr[j + L_start] = acc;

#pragma omp parallel for private(acc) schedule(dynamic)
        for (int s = j + 1; s < L.n; ++s) {
            acc = A.ptr[j + A.stride * s];
            for (int k = 0; k < j; ++k) {
                acc += L.ptr[k + L_start] * L.ptr[k + L.stride * s];
            }
            L.ptr[j + L.stride * s] = acc / L.ptr[j + L_start];
        }
    }
}

inline void extractBlock(const SubMatrix& block, SubMatrix& res) {
    for (int i = 0; i < res.n; ++i) {
        memcpy(&res.ptr[res.stride * i], &block.ptr[block.stride * i], res.m * sizeof(decltype(*block.ptr)));
    }
}

inline void storeBlock(SubMatrix& block, const SubMatrix& res) {
    for (int i = 0; i < res.n; ++i) {
        memcpy(&block.ptr[block.stride * i], &res.ptr[res.stride * i], res.m * sizeof(decltype(*block.ptr)));
    }
}

inline void storeBlockTransposed(SubMatrix& block, const SubMatrix& res) {
#pragma omp parallel for
    for (int i = 0; i < res.m; ++i) {
        for (int j = 0; j < res.n; ++j) {
            block.ptr[j + block.stride * i] = res.ptr[i + res.stride * j];
        }
    }
}

// X = ABT
inline void transposedMatMulRef(const SubMatrix& A, const SubMatrix& B, SubMatrix& X) {
#pragma omp parallel for
    for (int i = 0; i < A.n; ++i) {
        for (int j = 0; j < B.n; ++j) {
            for (int k = 0; k < A.m; ++k) {
                X.ptr[j + X.stride * i] += A.ptr[k + A.stride * i] * B.ptr[k + B.stride * j];
            }
        }
    }
}

inline void transposedMatMul(const SubMatrix& A, const SubMatrix& B, SubMatrix& X) {
    constexpr int pack_size = 4;
#pragma omp parallel for
    for (int i = 0; i < A.n; ++i) {
        auto X_start = X.stride * i;
        auto A_start = A.stride * i;
        for (int j = 0; j < B.n; ++j) {
            auto B_start = B.stride * j;

            __m256d sum = _mm256_setzero_pd();
            __m256d a, b;
            int idx = 0;
            for (; idx <= A.m - pack_size; idx += pack_size) {
                a = _mm256_loadu_pd(&A.ptr[idx + A_start]);
                b = _mm256_loadu_pd(&B.ptr[idx + B_start]);
                sum = _mm256_fmadd_pd(a, b, sum);
            }
            __m256i mask = _mm256_loadu_si256((const __m256i*)(maskarr + (pack_size - (A.m - idx))));
            __m256d a_tail = _mm256_maskload_pd(&A.ptr[idx + A_start], mask);
            __m256d b_tail = _mm256_maskload_pd(&B.ptr[idx + B_start], mask);
            sum = _mm256_fmadd_pd(a_tail, b_tail, sum);
            sum = _mm256_hadd_pd(sum, sum);
            __m128d hi = _mm256_extractf128_pd(sum, 1);
            __m128d res = _mm_add_sd(_mm256_castpd256_pd128(sum), hi);
            X.ptr[j + X_start] = _mm_cvtsd_f64(res);
        }
    }
}

// X = A - B
void matAdd(const SubMatrix& A, const SubMatrix& B, SubMatrix& X) {
    const int pack_size = 4;
    const int quot = A.m / pack_size;
    const int rem = A.m % pack_size;

#pragma omp parallel for
    for (int i = 0; i < A.n; ++i) {
        int pos = 0;
        int A_start = A.stride * i;
        int B_start = B.stride * i;
        int X_start = X.stride * i;

        __m256d res = _mm256_setzero_pd();
        for (int j = 0; j < quot; ++j) {
            pos = j * pack_size;
            __m256d a = _mm256_loadu_pd(&A.ptr[pos + A_start]);
            __m256d b = _mm256_loadu_pd(&B.ptr[pos + B_start]);
            res = _mm256_add_pd(a, b);
            _mm256_storeu_pd(&X.ptr[pos + X_start], res);
        }
        pos = quot * pack_size;
        __m256i mask = _mm256_loadu_si256((const __m256i*)(maskarr + (pack_size - rem)));
        __m256d a_tail = _mm256_maskload_pd(&A.ptr[pos + A_start], mask);
        __m256d b_tail = _mm256_maskload_pd(&B.ptr[pos + B_start], mask);
        res = _mm256_add_pd(a_tail, b_tail);
        _mm256_maskstore_pd(&X.ptr[pos + X_start], mask, res);
    }
}

// X = A - B
void matSub(const SubMatrix& A, const SubMatrix& B, SubMatrix& X) {
    const int pack_size = 4;
    const int quot = A.m / pack_size;
    const int rem = A.m % pack_size;

#pragma omp parallel for
    for (int i = 0; i < A.n; ++i) {
        int pos = 0;
        int A_start = A.stride * i;
        int B_start = B.stride * i;
        int X_start = X.stride * i;

        __m256d res = _mm256_setzero_pd();
        for (int j = 0; j < quot; ++j) {
            pos = j * pack_size;
            __m256d a = _mm256_loadu_pd(&A.ptr[pos + A_start]);
            __m256d b = _mm256_loadu_pd(&B.ptr[pos + B_start]);
            res = _mm256_sub_pd(a, b);
            _mm256_storeu_pd(&X.ptr[pos + X_start], res);
        }
        pos = quot * pack_size;
        __m256i mask = _mm256_loadu_si256((const __m256i*)(maskarr + (pack_size - rem)));
        __m256d a_tail = _mm256_maskload_pd(&A.ptr[pos + A_start], mask);
        __m256d b_tail = _mm256_maskload_pd(&B.ptr[pos + B_start], mask);
        res = _mm256_sub_pd(a_tail, b_tail);
        _mm256_maskstore_pd(&X.ptr[pos + X_start], mask, res);
    }
}

// X = A - B
void matSub_NOSIMD(const SubMatrix& A, const SubMatrix& B, SubMatrix& X) {
#pragma omp parallel for
    for (int i = 0; i < A.n; ++i) {
        for (int j = 0; j < A.m; ++j) {
            X.ptr[j + i * X.stride] = A.ptr[j + i * A.stride] - B.ptr[j + i * A.stride];
        }
    }
}

void blockedCholeskyDec(const SubMatrix& A, SubMatrix& L) {
    const int blk_mult = 512;
    const int blk_size = A.n < blk_mult ? A.n : blk_mult;
    const int blk_buffer_size = blk_size * blk_size;
    const int rem_blk_size = A.n - blk_size;
    const int rem_ll_buffer_size = rem_blk_size * blk_size;
    const int rem_lr_buffer_size = rem_blk_size * rem_blk_size;
    MatBuffer A_ul_buf(blk_buffer_size);
    SubMatrix A_ul{ A_ul_buf.data(), blk_size, blk_size, blk_size };
    SubMatrix A_ul_src{ A.ptr, blk_size, blk_size, A.stride };
    MatBuffer L_ul_buf(blk_buffer_size);
    SubMatrix L_ul{ L_ul_buf.data(), blk_size, blk_size, blk_size };

    extractBlock(A_ul_src, A_ul);
    choleskyDecomp(A_ul, L_ul);

    SubMatrix L_ul_dst{ L.ptr, blk_size, blk_size, L.stride };
    storeBlock(L_ul_dst, L_ul);

    if (rem_blk_size == 0) {
        return;
    }

    MatBuffer A_ll_buf(rem_ll_buffer_size);
    SubMatrix A_ll{ A_ll_buf.data(), rem_blk_size, blk_size, blk_size };
    SubMatrix A_ll_src{ &A.ptr[blk_size * A.stride], rem_blk_size, blk_size, A.stride };
    MatBuffer L_ll_buf(rem_ll_buffer_size);
    SubMatrix L_ll{ L_ll_buf.data(), rem_blk_size, blk_size, blk_size };

    extractBlock(A_ll_src, A_ll);
    solveLinearSystemTransposed(L_ul, A_ll, L_ll);
    SubMatrix L_ll_dst{ &L.ptr[blk_size * L.stride], rem_blk_size, blk_size, L.stride };
    storeBlock(L_ll_dst, L_ll);

    MatBuffer A_lr_buf(rem_lr_buffer_size);
    SubMatrix A_lr{ A_lr_buf.data(), rem_blk_size, rem_blk_size, rem_blk_size };
    SubMatrix A_lr_src{ &A.ptr[blk_size * A.stride + blk_size], rem_blk_size, rem_blk_size, A.stride };

    extractBlock(A_lr_src, A_lr);

    MatBuffer L_ll_prod_buf(rem_lr_buffer_size);
    SubMatrix L_ll_prod{ L_ll_prod_buf.data(), rem_blk_size, rem_blk_size, rem_blk_size };

    transposedMatMul(L_ll, L_ll, L_ll_prod);
    matSub(A_lr, L_ll_prod, A_lr);

    SubMatrix L_lr{ &L.ptr[blk_size * L.stride + blk_size], rem_blk_size, rem_blk_size, L.stride };
    blockedCholeskyDec(A_lr, L_lr);
}

void blockedCholeskyDec_NOSIMD(const SubMatrix& A, SubMatrix& L) {
    const int blk_mult = 512;
    const int blk_size = A.n < blk_mult ? A.n : blk_mult;
    const int blk_buffer_size = blk_size * blk_size;
    const int rem_blk_size = A.n - blk_size;
    const int rem_ll_buffer_size = rem_blk_size * blk_size;
    const int rem_lr_buffer_size = rem_blk_size * rem_blk_size;
    MatBuffer A_ul_buf(blk_buffer_size);
    SubMatrix A_ul{ A_ul_buf.data(), blk_size, blk_size, blk_size };
    SubMatrix A_ul_src{ A.ptr, blk_size, blk_size, A.stride };
    MatBuffer L_ul_buf(blk_buffer_size);
    SubMatrix L_ul{ L_ul_buf.data(), blk_size, blk_size, blk_size };

    extractBlock(A_ul_src, A_ul);
    choleskyDecomp_NOSIMD(A_ul, L_ul);

    SubMatrix L_ul_dst{ L.ptr, blk_size, blk_size, L.stride };
    storeBlock(L_ul_dst, L_ul);

    if (rem_blk_size == 0) {
        return;
    }

    MatBuffer A_ll_buf(rem_ll_buffer_size);
    SubMatrix A_ll{ A_ll_buf.data(), rem_blk_size, blk_size, blk_size };
    SubMatrix A_ll_src{ &A.ptr[blk_size * A.stride], rem_blk_size, blk_size, A.stride };
    MatBuffer L_ll_buf(rem_ll_buffer_size);
    SubMatrix L_ll{ L_ll_buf.data(), rem_blk_size, blk_size, blk_size };

    extractBlock(A_ll_src, A_ll);
    solveLinearSystemTransposed_NOSIMD(L_ul, A_ll, L_ll);
    SubMatrix L_ll_dst{ &L.ptr[blk_size * L.stride], rem_blk_size, blk_size, L.stride };
    storeBlock(L_ll_dst, L_ll);

    MatBuffer A_lr_buf(rem_lr_buffer_size);
    SubMatrix A_lr{ A_lr_buf.data(), rem_blk_size, rem_blk_size, rem_blk_size };
    SubMatrix A_lr_src{ &A.ptr[blk_size * A.stride + blk_size], rem_blk_size, rem_blk_size, A.stride };

    extractBlock(A_lr_src, A_lr);

    MatBuffer L_ll_prod_buf(rem_lr_buffer_size);
    SubMatrix L_ll_prod{ L_ll_prod_buf.data(), rem_blk_size, rem_blk_size, rem_blk_size };

    transposedMatMulRef(L_ll, L_ll, L_ll_prod);
    matSub(A_lr, L_ll_prod, A_lr);

    SubMatrix L_lr{ &L.ptr[blk_size * L.stride + blk_size], rem_blk_size, rem_blk_size, L.stride };
    blockedCholeskyDec_NOSIMD(A_lr, L_lr);
}

void Cholesky_Decomposition(double* A, double* L, int n) {
    SubMatrix sA{ A, n, n, n };
    SubMatrix sL{ L, n, n, n };

    blockedCholeskyDec(sA, sL);
}

//
// Helpers
//

void randomlyFillLowerTriang(SubMatrix& X) {
    std::random_device rd;
    std::mt19937 gen(time(0));
    double max = 1.5l, min = 1.l;
    std::uniform_int_distribution<int> dist(0, 100);

    for (size_t i = 0; i < X.m; i++) {
        for (size_t j = 0; j < i + 1; j++) {
            X.ptr[j + X.stride * i] = static_cast<double>(dist(gen)) / 100.l * (max - min) + min;
        }
    }
}

double calRelResidual(const SubMatrix& A, const SubMatrix& X) {
    // Cacl relative residual using Frobenius norm
    double acc = 0.l;
    double t;
    for (size_t i = 0; i < A.m; ++i) {
        for (size_t j = 0; j < A.n; ++j) {
            t = A.ptr[j + A.stride * i] - X.ptr[j + X.stride * i];
            if (std::isnan(t)) {
                t = 0.l;
            }
            acc += t * t;
        }
    }

    auto diff_norm = std::sqrt(acc);

    acc = 0.l;
    for (size_t i = 0; i < A.m; ++i) {
        for (size_t j = 0; j < A.n; ++j) {
            t = A.ptr[j + A.stride * i];
            acc += t * t;
        }
    }

    auto A_norm = std::sqrt(acc / A.m);

    if (A_norm == 0) {
        return 0;
    }

    return diff_norm / A_norm;
}

int main(int argc, char* argv[]) {
    int K = 129;
    int numThreads = 4;
    size_t numTests = 25;
    if (argc > 1) {
        if (argc < 4) {
            throw std::runtime_error("Invalid number of arguments");
        }
        K = std::stoi(argv[1]);
        numTests = std::stoll(argv[2]);
        numThreads = std::stoi(argv[3]);
    }
    omp_set_num_threads(numThreads);
    int N = K, M = K;
    int size = N * M;

    std::cout << "Prepare test buffers\n";
    MatBuffer L_buf(size);
    MatBuffer B_buf(size);
    MatBuffer X_buf(size);
    MatBuffer X_test_buf(size);
    MatBuffer Res_check_buf(size);
    L_buf.memset(0);
    B_buf.memset(0);
    X_buf.memset(0);
    X_test_buf.memset(0);
    Res_check_buf.memset(0);

    SubMatrix L = { L_buf.data(), N, N, N };
    SubMatrix B = { B_buf.data(), N, M, M };
    SubMatrix X = { X_buf.data(), N, M, M };
    SubMatrix X_test = { X_test_buf.data(), N, M, M };
    SubMatrix Res_check = { Res_check_buf.data(), N, M, M };

    std::cout << "Prepare test data\n";
    randomlyFillLowerTriang(X_test);
    transposedMatMul(X_test, X_test, B);

    std::cout << "Start comparsion...\n";

    testing::TestResults tRes(numTests);

    std::cout << "blockedCholeskyDec no avx v. blockedCholeskyDec avx\n";

    for (size_t i = 0; i < numTests; ++i) {
        tRes.set(i, testing::meassureCall(blockedCholeskyDec_NOSIMD, B, X), testing::meassureCall(blockedCholeskyDec, B, X));
    }

    tRes.finalize();

    auto min = tRes.getMin();
    auto max = tRes.getMax();
    auto median = tRes.getMedian();

    std::cout << "Comparsion Ended\n";
    std::cout << "[Test] Min speedup: " << min.speedUp << " %, ref time: " << testing::convertUsToMs(min.refTime)
              << " ms, test time: " << testing::convertUsToMs(min.testTime) << " ms\n"
              << "[Test] Max speedup: " << max.speedUp << " %, ref time: " << testing::convertUsToMs(max.refTime)
              << " ms, test time: " << testing::convertUsToMs(max.testTime) << " ms\n"
              << "[Test] Median speedup " << median.speedUp << " %, ref time: " << testing::convertUsToMs(median.refTime)
              << " ms, test time: " << testing::convertUsToMs(median.testTime) << " ms\n";

    std::vector<int> num_threads = { 1, 2, 3, 4, 5, 6 };

    std::cout << "Start individual perf testing...\n";
    std::cout << "Testing: blockedCholeskyDec (noavx)\n";

    auto time = testing::meassureCall(blockedCholeskyDec_NOSIMD, B, X) / cvNanosecSec;
    std::cout << "Warmup time (ms): " << time << '\n';

    std::vector<double> time_results(numTests);
    for (auto p : num_threads) {
        omp_set_num_threads(p);
        for (size_t i = 0; i < numTests; ++i) {
            time_results[i] = testing::meassureCall(blockedCholeskyDec_NOSIMD, B, X) / cvNanosecSec;
        }
        std::sort(time_results.begin(), time_results.end());
        auto median_time = time_results[numTests % 2 ? (numTests - 1) / 2 : numTests / 2];
        double time_single_th, eff, speedup;
        if (p == 1) {
            time_single_th = median_time;
        }

        speedup = time_single_th / median_time;
        eff = speedup / p * 100.l;
        std::cout << std::setw(10) << p << std::setw(15) << median_time << std::setw(15) << speedup << std::setw(19) << eff << "%" << '\n';
    }

    std::cout << "Testing: blockedCholeskyDec (avx)\n";

    time = testing::meassureCall(blockedCholeskyDec, B, X) / cvNanosecSec;
    std::cout << "Warmup time (ms): " << time << '\n';

    for (auto p : num_threads) {
        omp_set_num_threads(p);
        for (size_t i = 0; i < numTests; ++i) {
            time_results[i] = testing::meassureCall(blockedCholeskyDec, B, X) / cvNanosecSec;
        }
        std::sort(time_results.begin(), time_results.end());
        auto median_time = time_results[numTests % 2 ? (numTests - 1) / 2 : numTests / 2];
        double time_single_th, eff, speedup;
        if (p == 1) {
            time_single_th = median_time;
        }

        speedup = time_single_th / median_time;
        eff = speedup / p * 100.l;
        std::cout << std::setw(10) << p << std::setw(15) << median_time << std::setw(15) << speedup << std::setw(19) << eff << "%" << '\n';
    }

    return 0;
}