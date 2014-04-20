#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cassert>
#include <cstdlib>
#include <ctime>
#include <cstdio>
#include <iostream>

using namespace std;

#define M_MAX 256
#define K_MAX 32768
#define MAT_MAX (1 << 21)

/*
 *         K                  N
 *    ____________       ____________
 *   |            |     |            |
 *   |            |     |            |
 *   |   M_{1}    |     |    M_{2}   |
 * M |            | * K |            |
 *   |            |     |            |
 *   |____________|     |____________|
 *
 */

int main() {
    cublasHandle_t cublasHandle;
    void* devPtr[3], * hostPtr[3];
    float alpha, beta;
    clock_t startTime;
    alpha = beta = 1.0;

    assert(cublasCreate(&cublasHandle) == CUBLAS_STATUS_SUCCESS);
    assert(cudaMalloc(&devPtr[0], M_MAX * K_MAX * sizeof(float)) == cudaSuccess);
    assert(hostPtr[0] = malloc(M_MAX * K_MAX * sizeof(float)));

    for (int m = 64; m <= M_MAX; m <<= 1) {
        for (int k = 64; k <= K_MAX; k <<= 1) {
            printf("M: %d, K: %d\n", m, k);
            int nMax;
            if (m < k) {
                nMax = MAT_MAX / k;
            } else {
                nMax = MAT_MAX / m;
            }
            assert(cudaMalloc(&devPtr[1], K_MAX * nMax * sizeof(float)) == cudaSuccess);
            assert(cudaMalloc(&devPtr[2], M_MAX * nMax * sizeof(float)) == cudaSuccess);
            assert(hostPtr[1] = malloc(K_MAX * nMax * sizeof(float)));
            assert(hostPtr[2] = malloc(M_MAX * nMax * sizeof(float)));
            for (int n = 128; n <= nMax; n <<= 1) {
                int repeat = (M_MAX / m) * (MAT_MAX / 64 / n) * (K_MAX / k);
                startTime = clock();
                while (repeat--) {
                    assert(cublasSetMatrix(k, m, sizeof(float), hostPtr[0], k, devPtr[0], k) == CUBLAS_STATUS_SUCCESS);
                    assert(cublasSetMatrix(n, k, sizeof(float), hostPtr[1], n, devPtr[1], n) == CUBLAS_STATUS_SUCCESS);
                    assert(cublasSetMatrix(n, m, sizeof(float), hostPtr[2], n, devPtr[2], n) == CUBLAS_STATUS_SUCCESS);
                    assert(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, (const float*) devPtr[1], n, (const float*) devPtr[0], k, &beta, (float*) devPtr[2], n) == CUBLAS_STATUS_SUCCESS);
                    assert(cublasGetMatrix(k, m, sizeof(float), devPtr[0], k, hostPtr[0], k) == CUBLAS_STATUS_SUCCESS);
                    assert(cublasGetMatrix(n, k, sizeof(float), devPtr[1], n, hostPtr[1], n) == CUBLAS_STATUS_SUCCESS);
                    assert(cublasGetMatrix(n, m, sizeof(float), devPtr[2], n, hostPtr[2], n) == CUBLAS_STATUS_SUCCESS);
                }
                printf("N: %d, time: %lf\n", n, (double) (clock() - startTime) / CLOCKS_PER_SEC);
            }
            free(hostPtr[2]);
            free(hostPtr[1]);
            assert(cudaFree(devPtr[2]) == cudaSuccess);
            assert(cudaFree(devPtr[1]) == cudaSuccess);
        }
    }

    free(hostPtr[0]);
    assert(cudaFree(devPtr[0]) == cudaSuccess);
    cublasDestroy(cublasHandle);
    return 0;
}

