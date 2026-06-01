#include <stdio.h>
#include <complex>
#include <windows.h>
#include <cblas.h>
#include <random>
#include <cmath>
#include <omp.h>
#include <openblas_config.h>

using namespace std;

using Complex = complex<double>;

int main() {
    SetConsoleCP(1251);
    SetConsoleOutputCP(1251);

    int num_cores = omp_get_max_threads();

    openblas_set_num_threads(num_cores);

    LARGE_INTEGER freq, start, end;

    const int N = 2048;

    printf("Умножение матриц: %dx%d (double complex)\n", N, N);
    printf("Теоретическая сложность: 2 * %d^3 = %.2e операций\n\n",
        N, 2.0 * N * N * N);

    Complex* A = new Complex[N * N];
    Complex* B = new Complex[N * N];
    Complex* C1 = new Complex[N * N];
    Complex* C2 = new Complex[N * N];
    Complex* C3 = new Complex[N * N];

    printf("Генерация случайных матриц A и B (%d x %d)\n", N, N);

    random_device rd;
    mt19937 gen(rd());

    uniform_real_distribution<double> dis(-1.0, 1.0);

    for (int i = 0; i < N * N; ++i) {

        A[i] = Complex(dis(gen), dis(gen));
        B[i] = Complex(dis(gen), dis(gen));
        C1[i] = Complex(0.0, 0.0);
        C2[i] = Complex(0.0, 0.0);
        C3[i] = Complex(0.0, 0.0);
    }

    QueryPerformanceFrequency(&freq);

    double ops = 2.0 * N * N * N;

    printf("1-й вариант: Классический алгоритм\n");

    QueryPerformanceCounter(&start);

    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            Complex aik = A[i * N + k];
            for (int j = 0; j < N; ++j) {
                C1[i * N + j] += aik * B[k * N + j];
            }
        }
    }

    QueryPerformanceCounter(&end);

    double time1 = static_cast<double>(end.QuadPart - start.QuadPart) / freq.QuadPart;

    double mflops1 = ops / time1 / 1e6;

    printf("Время выполнения: %.3f сек\n", time1);
    printf("Производительность: %.2f MFLOPS\n\n", mflops1);

    printf("2-й вариант: BLAS\n");

    double alpha[] = { 1.0, 0.0 };
    double beta[] = { 0.0, 0.0 };

    QueryPerformanceCounter(&start);

    cblas_zgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        N,
        N,
        N,
        alpha,
        static_cast<void*>(A),
        N,
        static_cast<void*>(B),
        N,
        beta,
        static_cast<void*>(C2),
        N
    );

    QueryPerformanceCounter(&end);

    double time2 = static_cast<double>(end.QuadPart - start.QuadPart) / freq.QuadPart;

    double mflops2 = ops / time2 / 1e6;

    printf("Время выполнения: %.3f сек\n", time2);
    printf("Производительность: %.2f MFLOPS\n\n", mflops2);

    printf("3-й вариант: Блочный алгоритм\n");

    QueryPerformanceCounter(&start);

    const int BS = 64;

#pragma omp parallel for schedule(static)
    for (int ii = 0; ii < N; ii += BS) {

        for (int kk = 0; kk < N; kk += BS) {

            for (int jj = 0; jj < N; jj += BS) {

                for (int i = ii; i < ii + BS && i < N; ++i) {

                    for (int k = kk; k < kk + BS && k < N; ++k) {

                        Complex aik = A[i * N + k];

                        for (int j = jj; j < jj + BS && j < N; ++j) {

                            C3[i * N + j] += aik * B[k * N + j];
                        }
                    }
                }
            }
        }
    }

    QueryPerformanceCounter(&end);

    double time3 =
        static_cast<double>(end.QuadPart - start.QuadPart) / freq.QuadPart;

    double mflops3 = ops / time3 / 1e6;

    double percent_of_blas = (mflops3 / mflops2) * 100.0;

    printf("Время выполнения: %.3f сек\n", time3);
    printf("Производительность: %.2f MFLOPS\n", mflops3);

    printf("Производительность относительно BLAS: %.1f%%\n\n", percent_of_blas);

    printf("Проверка точности:\n");

    Complex* Delta = new Complex[N * N];

    double minus_one[] = { -1.0, 0.0 };

    cblas_zcopy(
        N * N,
        static_cast<void*>(C2),
        1,
        static_cast<void*>(Delta),
        1
    );

    cblas_zaxpy(
        N * N,
        minus_one,
        static_cast<void*>(C1),
        1,
        static_cast<void*>(Delta),
        1
    );

    double norm = cblas_dznrm2(
            N * N,
            static_cast<void*>(Delta),
            1
        );

    printf("Норма разности (Классический - BLAS): %.2e\n", norm);

    cblas_zcopy(
        N * N,
        static_cast<void*>(C2),
        1,
        static_cast<void*>(Delta),
        1
    );

    cblas_zaxpy(
        N * N,
        minus_one,
        static_cast<void*>(C3),
        1,
        static_cast<void*>(Delta),
        1
    );

    norm = cblas_dznrm2(
            N * N,
            static_cast<void*>(Delta),
            1
        );

    printf("Норма разности (Новый способ - BLAS): %.2e\n", norm);

    if (percent_of_blas >= 30.0) {
        printf("Требование выполнено!\n\n");
    }
    else {
        printf("Скорость ниже 30%% от BLAS.\n\n");
    }

    printf("ФИО: Павленко Кирилл Дмитриевич\n");
    printf("Группа: 020303-АИСа-о25\n");


    delete[] Delta;
    delete[] A;
    delete[] B;
    delete[] C1;
    delete[] C2;
    delete[] C3;

    return 0;
}