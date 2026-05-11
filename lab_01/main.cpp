#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <immintrin.h>
#include <omp.h>
#include <stdlib.h>

static constexpr int64_t maskarr[] = {
    -1, -1, -1, -1,
    0, 0, 0, 0
};

struct SubMatrix {
    double* ptr;
    int n;
    int m;
    int stride;
};

class MatBuffer {
public:
    explicit MatBuffer(size_t size, size_t alignment = 32) : _size(size), _align (alignment) {
        _buffer = _aligned_malloc(_size * sizeof(double), _align);
        memset(_buffer, 0, _size * sizeof(double));
    }
    ~MatBuffer() {
        _aligned_free(_buffer);
    }

    MatBuffer(const MatBuffer&) = delete;
    MatBuffer& operator=(const MatBuffer&) = delete;

    double* data() {
        return static_cast<double*>(_buffer);
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
                        __m256d A_elems = _mm256_load_pd(&A.ptr[pos + A_start]);
                        __m256d X_elems = _mm256_load_pd(&X.ptr[pos + X_start]);
                        sum = _mm256_fmadd_pd(A_elems, X_elems, sum);
                    }
                    pos = j - rem;

                    __m256i mask = _mm256_load_si256((const __m256i*)(maskarr + (packSize - rem)));
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

inline void choleskyDecomp(const SubMatrix& A, SubMatrix& L) {
    int packSize = 4;
    for (int j = 0; j < A.m; ++j) {
        int L_start =  L.stride * j;
        double acc = A.ptr[j + A.stride * j];
        int quot = j / packSize;
        int rem = j % packSize;

        __m256d sum = _mm256_setzero_pd();
        __m256d curr_val = _mm256_setzero_pd();
        for (int s = 0; s < quot; ++s) { 
            curr_val = _mm256_load_pd(&L.ptr[s * packSize + L_start]);
            sum = _mm256_fmadd_pd(curr_val, curr_val, sum);
        }
        __m256i mask = _mm256_load_si256((const __m256i*)(maskarr + (packSize - rem)));
        curr_val = _mm256_maskload_pd(&L.ptr[quot * packSize + L_start], mask);
        sum = _mm256_fmadd_pd(curr_val, curr_val, sum);

        sum = _mm256_hadd_pd(sum, sum);
        __m128d hi = _mm256_extractf128_pd(sum, 1);
        __m128d res = _mm_add_sd(_mm256_castpd256_pd128(sum), hi);
        acc -= _mm_cvtsd_f64(res);

        acc = std::sqrt(acc);
        L.ptr[j + L_start] = acc;

        #pragma omp parallel for private(acc, quot, rem, sum)
        for (int s = j + 1; s < L.n; ++s) {
            acc = A.ptr[j + A.stride * s];

            quot = j / packSize;
            rem = j % packSize;           
            sum = _mm256_setzero_pd();
            for (int k = 0; k < quot; ++k) { 
                __m256d a = _mm256_load_pd(&L.ptr[k * packSize + L_start]);
                __m256d b = _mm256_load_pd(&L.ptr[k * packSize + L.stride * s]);
                sum = _mm256_fmadd_pd(a, b, sum);
            }
            __m256i mask = _mm256_load_si256((const __m256i*)(maskarr + (packSize - rem)));
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

// Not tested yet
inline void matmul_ker_3_16(const double* A, const double* B, double* X, const int M, const int lda, const int ldx) {
    const double* Aptr = A;
    const double* Bptr = B;
    double* Xptr = X;

    __m256d c_11 = _mm256_setzero_pd();
    __m256d c_12 = _mm256_setzero_pd();
    __m256d c_13 = _mm256_setzero_pd();
    __m256d c_14 = _mm256_setzero_pd();

    __m256d c_21 = _mm256_setzero_pd();
    __m256d c_22 = _mm256_setzero_pd();
    __m256d c_23 = _mm256_setzero_pd();
    __m256d c_24 = _mm256_setzero_pd();

    __m256d c_31 = _mm256_setzero_pd();
    __m256d c_32 = _mm256_setzero_pd();
    __m256d c_33 = _mm256_setzero_pd();
    __m256d c_34 = _mm256_setzero_pd();

    const int offset[3] = {
        0, lda, 2 * lda
    };

    __m256d a_1, a_2, a_3, b_1;
    for (int s = 0; s < M; ++s) {
        a_1 = _mm256_set1_pd(Aptr[offset[0]]);
        a_2 = _mm256_set1_pd(Aptr[offset[1]]);
        a_3 = _mm256_set1_pd(Aptr[offset[2]]);

        b_1 = _mm256_loadu_pd(Bptr);
        c_11 = _mm256_fmadd_pd(a_1, b_1, c_11);
        c_21 = _mm256_fmadd_pd(a_2, b_1, c_12);
        c_31 = _mm256_fmadd_pd(a_3, b_1, c_21);

        b_1 = _mm256_loadu_pd(&Bptr[offset[1]]);
        c_12 = _mm256_fmadd_pd(a_1, b_1, c_11);
        c_22 = _mm256_fmadd_pd(a_2, b_1, c_12);
        c_33 = _mm256_fmadd_pd(a_3, b_1, c_21);

        b_1 = _mm256_loadu_pd(&Bptr[offset[2]]);
        c_13 = _mm256_fmadd_pd(a_1, b_1, c_11);
        c_23 = _mm256_fmadd_pd(a_2, b_1, c_12);
        c_33 = _mm256_fmadd_pd(a_3, b_1, c_21);

        
        Aptr += 1;
        Bptr += +1;
    }

    _mm256_store_pd(X, c_11);
    _mm256_store_pd(X + 4, c_12);
    _mm256_store_pd(X + 8, c_13);

    _mm256_store_pd(X + ldx, c_21);
    _mm256_store_pd(X + ldx + 4, c_22);
    _mm256_store_pd(X + ldx + 8, c_23);

    _mm256_store_pd(X + 2 * ldx, c_31);
    _mm256_store_pd(X + 2 * ldx + 4, c_32);
    _mm256_store_pd(X + 2 * ldx + 8, c_33);
}

inline void transposedMatMul(const SubMatrix& A, const SubMatrix& B, SubMatrix& X) {
    int pack_size = 4;
    #pragma omp parallel for
    for (int i = 0; i < A.n; ++i) {
        for (int j = 0; j < B.n; ++j) {
            int A_start = A.stride * i;
            int B_start = B.stride * j;
            int quot = A.m / pack_size;
            int rem = A.m % pack_size; 
            int pos = 0;
            for (int k = 0; k < A.m; ++k) {
                X.ptr[j + X.stride * i] += A.ptr[k + A.stride * i] * B.ptr[k + B.stride * j];
            }
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
            __m256d a = _mm256_load_pd(&A.ptr[pos + A_start]);
            __m256d b = _mm256_load_pd(&B.ptr[pos + B_start]);
            res = _mm256_add_pd(a, b);
            _mm256_store_pd(&X.ptr[pos + X_start], res);
        }
        pos = quot * pack_size;
        __m256i mask = _mm256_load_si256((const __m256i*)(maskarr + (pack_size - rem)));
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
            __m256d a = _mm256_load_pd(&A.ptr[pos + A_start]);
            __m256d b = _mm256_load_pd(&B.ptr[pos + B_start]);
            res = _mm256_sub_pd(a, b);
            _mm256_store_pd(&X.ptr[pos + X_start], res);
        }
        pos = quot * pack_size;
        __m256i mask = _mm256_load_si256((const __m256i*)(maskarr + (pack_size - rem)));
        __m256d a_tail = _mm256_maskload_pd(&A.ptr[pos + A_start], mask);
        __m256d b_tail = _mm256_maskload_pd(&B.ptr[pos + B_start], mask);
        res = _mm256_sub_pd(a_tail, b_tail);
        _mm256_maskstore_pd(&X.ptr[pos + X_start], mask, res);

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
    SubMatrix A_ul{A_ul_buf.data(), blk_size, blk_size, blk_size};
    SubMatrix A_ul_src{A.ptr, blk_size, blk_size, A.stride};
    MatBuffer L_ul_buf(blk_buffer_size);
    SubMatrix L_ul{L_ul_buf.data(), blk_size, blk_size, blk_size};

    extractBlock(A_ul_src, A_ul);
    choleskyDecomp(A_ul, L_ul);

    SubMatrix L_ul_dst{L.ptr, blk_size, blk_size, L.stride};
    storeBlock(L_ul_dst, L_ul);

    if (rem_blk_size == 0) {
        return;
    }


    MatBuffer A_ll_buf(rem_ll_buffer_size);
    SubMatrix A_ll{A_ll_buf.data(), rem_blk_size, blk_size, blk_size};
    SubMatrix A_ll_src{&A.ptr[blk_size * A.stride], rem_blk_size, blk_size, A.stride};
    MatBuffer L_ll_buf(rem_ll_buffer_size);
    SubMatrix L_ll{L_ll_buf.data(), rem_blk_size, blk_size, blk_size};

    extractBlock(A_ll_src, A_ll);
    solveLinearSystemTransposed(L_ul, A_ll, L_ll);
    SubMatrix L_ll_dst{&L.ptr[blk_size * L.stride], rem_blk_size, blk_size, L.stride};
    storeBlock(L_ll_dst, L_ll);

    MatBuffer A_lr_buf(rem_lr_buffer_size);
    SubMatrix A_lr{A_lr_buf.data(), rem_blk_size, rem_blk_size, rem_blk_size};
    SubMatrix A_lr_src{&A.ptr[blk_size * A.stride + blk_size], rem_blk_size, rem_blk_size, A.stride};

    extractBlock(A_lr_src, A_lr);

    MatBuffer L_ll_prod_buf(rem_lr_buffer_size);
    SubMatrix L_ll_prod{L_ll_prod_buf.data(), rem_blk_size, rem_blk_size, rem_blk_size};

    transposedMatMul(L_ll, L_ll, L_ll_prod);
    matSub(A_lr, L_ll_prod, A_lr);

    SubMatrix L_lr{&L.ptr[blk_size * L.stride + blk_size], rem_blk_size, rem_blk_size, L.stride};
    blockedCholeskyDec(A_lr, L_lr);
}

void Cholesky_Decomposition(double * A, double * L, int n) {
    SubMatrix sA{A, n, n, n};
    SubMatrix sL{L, n, n, n};

    blockedCholeskyDec(sA, sL);
}

//
// Helpers
//

void randomlyFillLowerTriang(SubMatrix& X) {
    std::random_device rd;
    std::mt19937 gen(time(0));
    std::uniform_int_distribution<int> dist(1, 30);

    for (size_t i = 0; i < X.m; i++) {
        for (size_t j = 0; j < i + 1; j++) {
            X.ptr[j + X.stride * i] = dist(gen);
        }
    }
}

void randomlyFill(SubMatrix& X) {
    std::random_device rd;
    std::mt19937 gen(time(0));
    std::uniform_int_distribution<int> dist(0, 30);

    for (size_t i = 0; i < X.m; i++) {
        for (size_t j = 0; j < X.n; j++) {
            X.ptr[j + X.stride * i] = dist(gen);
        }
    }
}

void printMatrix(SubMatrix& X) {
    for(int i = 0; i < X.m; ++i) {
        for (int j = 0; j < X.n; ++j) {
            std::cout << X.ptr[j + X.stride * i] << ' ';
        }
        std::cout << '\n';
    }
}

bool compareEltWise(const SubMatrix& A, const SubMatrix& B, double eps = 1e-5) {
    for (size_t i = 0; i < A.n; ++i) {
        for (size_t j = 0; j < A.m; ++j) {
            if (std::abs(A.ptr[j + A.stride * i] - B.ptr[j + B.stride * i]) >= eps) {
                return false;
            }
        }
    }
    
    return true;
}

int main() {
    omp_set_num_threads(6);
    int K = 2000;
    int N = K, M = K;
    int size = N * M;

    MatBuffer L_buf(size);
    MatBuffer B_buf(size, 1);
    MatBuffer X_buf(size, 1);
    MatBuffer X_test_buf(size, 1);
    memset(X_buf.data(), 0, size * sizeof(double));
    memset(X_test_buf.data(), 0, size * sizeof(double));
    memset(L_buf.data(), 0, size * sizeof(double));
    memset(B_buf.data(), 0, size * sizeof(double));

    SubMatrix L = {L_buf.data(), N, N, N};
    SubMatrix B = {B_buf.data(), N, M, M};
    SubMatrix X = {X_buf.data(), N, M, M};
    SubMatrix X_test = {X_test_buf.data(), N, M, M};

    randomlyFillLowerTriang(X_test);
    transposedMatMul(X_test, X_test, B);
    
    auto timeStart = std::chrono::steady_clock::now();
    // choleskyDecomp(B, X);
    transposedMatMulRef(X_test, X_test, B);
    auto timeRef = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - timeStart).count();
    // std::cout << "[Ref] " << (compareEltWise(X_test, X) ? "Test passed" : "Test failed!") << '\n';
    std::cout << "[Ref] " << timeRef << " ms\n";

    memset(X_buf.data(), 0, size * sizeof(double));

    timeStart = std::chrono::steady_clock::now();
    // Cholesky_Decomposition(B.ptr, X.ptr, N);
    transposedMatMul(X_test, X_test, B);
    timeRef = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - timeStart).count();
    // std::cout << "[Test] " << (compareEltWise(X_test, X) ? "Test passed" : "Test failed!") << '\n';
    std::cout << "[Test] " << timeRef << " ms\n";


    // printMatrix(X);

    std::cout << '\n';

    // printMatrix(X_test);

    // printMatrix(X_test);

    // std::vector<double> exB_test_vec(N * M / 4);
    // memset(exB_test_vec.data(), 0, exB_test_vec.size() * sizeof(double));
    // SubMatrix exB_test = {exB_test_vec.data(), N / 2, M / 2, M / 2};
    // SubMatrix src = {B.ptr + M / 2, N / 2, M / 2, B.stride};
    // extractBlock(src, exB_test);

    // timeStart = std::chrono::steady_clock::now();
    // MatBuffer buffer(N * M / 4);
    // SubMatrix exB_test = {buffer.data(), N / 2, M / 2, M / 2};
    // extractBlock(src, exB_test);
    // timeRef = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - timeStart).count();
    // std::cout << "[extract] " << timeRef << " ms\n";



    
    return 0;
}