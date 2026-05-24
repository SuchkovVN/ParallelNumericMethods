#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <random>
#include <iomanip>
#include <chrono>

struct CRSMatrix {
    int n; 
    int m; 
    int nz; 
    std::vector<double> val;
    std::vector<int> colIndex;
    std::vector<int> rowPtr;
};


void SLE_Solver_CRS(CRSMatrix & A, double * b, double eps, int max_iter, double * x, int & count)
{
    int n = A.n;
    count = 0;

    std::vector<double> r(n, 0.0);
    std::vector<double> p(n, 0.0);
    std::vector<double> Ap(n, 0.0);

    int max_threads = omp_get_max_threads();
    std::vector<std::vector<double>> local_Ap(max_threads, std::vector<double>(n, 0.0));

    double norm_b_sq = 0.0;

    #pragma omp parallel for reduction(+:norm_b_sq)
    for (int i = 0; i < n; i++) {
        x[i] = 0.0;
        r[i] = b[i];
        p[i] = b[i];
        norm_b_sq += b[i] * b[i];
    }

    double norm_b = sqrt(norm_b_sq);
    
    if (norm_b == 0.0) {
        return;
    }

    double rr = norm_b_sq;

    for (int iter = 0; iter < max_iter; iter++) {
        
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int n_threads = omp_get_num_threads();

            for (int i = 0; i < n; i++) {
                local_Ap[tid][i] = 0.0;
            }

            #pragma omp for
            for (int i = 0; i < n; i++) {
                double sum = 0.0;
                for (int k = A.rowPtr[i]; k < A.rowPtr[i+1]; k++) {
                    int j = A.colIndex[k];
                    double v = A.val[k];
                    
                    sum += v * p[j];
                    
                    if (i != j) {
                        local_Ap[tid][j] += v * p[i];
                    }
                }
                local_Ap[tid][i] += sum;
            } 

            #pragma omp for
            for (int i = 0; i < n; i++) {
                double total = 0.0;
                for (int t = 0; t < n_threads; t++) {
                    total += local_Ap[t][i];
                }
                Ap[i] = total;
            }
        } 

        double pAp = 0.0;
        #pragma omp parallel for reduction(+:pAp)
        for (int i = 0; i < n; i++) {
            pAp += p[i] * Ap[i];
        }

        double alpha = rr / pAp;

        double dx_norm_sq = 0.0;
        #pragma omp parallel for reduction(+:dx_norm_sq)
        for (int i = 0; i < n; i++) {
            double dx = alpha * p[i];
            x[i] += dx;
            dx_norm_sq += dx * dx;
        }

        count++;

        double dx_norm = sqrt(dx_norm_sq);
        if (dx_norm / norm_b < eps) {
            break;
        }

        double rr_new = 0.0;
        #pragma omp parallel for reduction(+:rr_new)
        for (int i = 0; i < n; i++) {
            r[i] -= alpha * Ap[i];
            rr_new += r[i] * r[i];
        }

        double beta = rr_new / rr;

        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            p[i] = r[i] + beta * p[i];
        }

        rr = rr_new;
    }
}

void multiply_CRS_vector(const CRSMatrix& A, const std::vector<double>& x, std::vector<double>& res) {
    int n = A.n;
    fill(res.begin(), res.end(), 0.0);
    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        for (int k = A.rowPtr[i]; k < A.rowPtr[i+1]; k++) {
            int j = A.colIndex[k];
            double v = A.val[k];
            sum += v * x[j];
            if (i != j) {
                res[j] += v * x[i];
            }
        }
        res[i] += sum;
    }
}


CRSMatrix generate_spd_crs(int n, int max_nz_per_row) {
    CRSMatrix A;
    A.n = n;
    A.m = n;
    A.rowPtr.push_back(0);
    A.nz = 0;

    std::mt19937 gen(144); 
    std::uniform_real_distribution<double> dist_val(-10.0, 10.0);
    std::uniform_int_distribution<int> dist_col(0, n - 1);

    std::vector<double> diag_vals(n, 0.0);
    std::vector<std::vector<std::pair<int, double>>> rows(n);

    for (int i = 0; i < n; i++) {
        rows[i].push_back({i, 0.0}); 
        
        int nz_count = dist_col(gen) % max_nz_per_row;
        for (int k = 0; k < nz_count; k++) {
            int j = dist_col(gen);
            if (j > i) {
                double val = dist_val(gen);
                rows[i].push_back({j, val});
                diag_vals[i] += abs(val);
                diag_vals[j] += abs(val);
            }
        }
    }

    for (int i = 0; i < n; i++) {
        rows[i][0].second = diag_vals[i] + 10.0 + abs(dist_val(gen));
        
        for (auto& elem : rows[i]) {
            A.colIndex.push_back(elem.first);
            A.val.push_back(elem.second);
            A.nz++;
        }
        A.rowPtr.push_back(A.nz);
    }
    return A;
}

double calc_matrix_2norm(const CRSMatrix& A) {
    int n = A.n;
    std::vector<double> q(n, 1.0 / sqrt(n));
    std::vector<double> z(n, 0.0);
    double lambda = 0.0;

    for (int iter = 0; iter < 100; iter++) {
        multiply_CRS_vector(A, q, z);
        double z_norm = 0.0;
        for (int i = 0; i < n; i++) z_norm += z[i] * z[i];
        z_norm = sqrt(z_norm);
        
        double new_lambda = 0.0;
        for (int i = 0; i < n; i++) {
            q[i] = z[i] / z_norm;
            new_lambda += q[i] * z[i]; // (Aq, q)
        }
        if (abs(new_lambda - lambda) < 1e-6) break;
        lambda = new_lambda;
    }
    return lambda;
}

double norm_2(const std::vector<double>& v) {
    double sum = 0.0;
    for (double val : v) sum += val * val;
    return sqrt(sum);
}

void test_correctness() {
    std::cout << "--- Test begin ---" << '\n';
    std::vector<int> test_sizes = {10, 50, 100, 500, 1000};
    
    for (int n : test_sizes) {
        CRSMatrix A = generate_spd_crs(n, 10);
        
        std::vector<double> x_true(n);
        std::mt19937 gen(123);
        std::uniform_real_distribution<double> dist_x(-1.0, 1.0);
        for (int i = 0; i < n; i++) x_true[i] = dist_x(gen);
        
        std::vector<double> b(n);
        multiply_CRS_vector(A, x_true, b);
        
        std::vector<double> x(n, 0.0);
        int iters = 0;
        
        SLE_Solver_CRS(A, b.data(), 1e-10, 10000, x.data(), iters);
        
        std::vector<double> Ax(n);
        multiply_CRS_vector(A, x, Ax);
        
        std::vector<double> residual(n);
        for (int i = 0; i < n; i++) residual[i] = Ax[i] - b[i];
        
        double norm_res = norm_2(residual);
        double norm_A = calc_matrix_2norm(A);
        double rel_error = norm_res / norm_A;
        
        std::cout << "n = " <<std::setw(5) << n 
             << " | Num inters: " << std::setw(4) << iters 
             << " | Rel. error: " << std::scientific << rel_error;
             
        if (rel_error < 1e-8) std::cout << " [PASSED]" << '\n';
        else std::cout << " [FAILED]" << '\n';
    }
    std::cout << '\n';
}

void test_performance() {
    std::cout << "--- Perf test begin ---" << '\n';
    int n = 100000; 
    int max_nz_per_row = 100;
    std::cout << " mat size " << n << "x" << n << "..." << std::flush;
    
    CRSMatrix A = generate_spd_crs(n, max_nz_per_row);
    std::cout << " A.nz: " << A.nz << '\n';
    
    std::vector<double> x_true(n, 1.0);
    std::vector<double> b(n);
    multiply_CRS_vector(A, x_true, b);
    std::vector<double> x(n, 0.0);
    
    std::vector<int> threads = {1, 2, 3, 4, 5, 6};
    double time_seq = 0.0;
    
    std::cout << std::fixed << std::setprecision(4);
    
    for (int p : threads) {
        omp_set_num_threads(p);
        fill(x.begin(), x.end(), 0.0);
        int iters = 0;
        auto start = std::chrono::steady_clock::now();
        SLE_Solver_CRS(A, b.data(), 1e-8, 1000, x.data(), iters);
        double time_spent = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count() / 1'000.l;

        if (p == 1) {
            time_seq = time_spent;
        }      
        double speedup = time_seq / time_spent;
        double efficiency = (speedup / p) * 100.0;

        std::cout << std::setw(10) << p << std::setw(15) << time_spent 
            << std::setw(15) << speedup << std::setw(19) << efficiency << "%" << '\n';
        
    }
}

int main() {
    test_correctness();
    test_performance();
    return 0;
}