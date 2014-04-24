#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cassert>
#include <cstdlib>
#include <ctime>
#include <cstdio>
#include <iostream>
#include <sys/time.h>

using namespace std;

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
    void* devPtr, * hostPtr;
    float alpha, beta;
    int mCandidate[] = {96, 256, 384};
    int kCandidate[] = {363, 2304, 2400, 3456};
    clock_t startTime;
    struct timeval tv1, tv2;
    alpha = beta = 1.0;
    assert(cublasCreate(&cublasHandle) == CUBLAS_STATUS_SUCCESS);
    assert(cudaMalloc(&devPtr, (1 << 30)) == cudaSuccess);
    assert(hostPtr = malloc(1 << 30));

#if 0
    int m = mCandidate[2];
    int k = kCandidate[3];
    int n = 1024;
    gettimeofday(&tv1, 0);

    cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, (const float*) devPtr, n, (const float*) devPtr, k, &beta, (float*) devPtr, n);
    cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, (const float*) devPtr, n, (const float*) devPtr, k, &beta, (float*) devPtr, n);
    cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, (const float*) devPtr, n, (const float*) devPtr, k, &beta, (float*) devPtr, n);
    cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, (const float*) devPtr, n, (const float*) devPtr, k, &beta, (float*) devPtr, n);
    assert(cublasGetMatrix(1, 1, sizeof(float), devPtr, 1, hostPtr, 1) == CUBLAS_STATUS_SUCCESS);

    gettimeofday(&tv2, 0);
    printf("For n = %d, time: %lf\n", n, (double) (tv2.tv_sec - tv1.tv_sec) * 1000000 + tv2.tv_usec - tv1.tv_usec);
#else
    for (int mIdx = 0; mIdx < 3; ++mIdx) {
        for (int kIdx = 0; kIdx < 4; ++kIdx) {
            int m = mCandidate[mIdx];
            int k = kCandidate[kIdx];
            printf("Using m = %d, k = %d\n", m, k);
            int nMax = (1 << 28) / k;
            int nMaxRounded;
            asm("bsrl %1, %0"
                    : "=r" (nMaxRounded)
                    : "r" (nMax));
            for (int n = 512; n <= (1 << nMaxRounded); n <<= 1) {
                int repeat = ((unsigned long long ) 1 << 39) / m / k / n;
                printf("Repeating for %d times\n", repeat);
                gettimeofday(&tv1, 0);
                while (repeat--) {
                    assert(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, (const float*) devPtr, n, (const float*) devPtr, k, &beta, (float*) devPtr, n) == CUBLAS_STATUS_SUCCESS);
                }
                assert(cublasGetMatrix(128, 128, sizeof(float), devPtr, 128, hostPtr, 128) == CUBLAS_STATUS_SUCCESS);
                gettimeofday(&tv2, 0);
                printf("For n = %d, time: %lf\n", n, (double) (tv2.tv_sec - tv1.tv_sec) * 1000000 + tv2.tv_usec - tv1.tv_usec);
            }
        }
    }
#endif
    free(hostPtr);
    assert(cudaFree(devPtr) == cudaSuccess);
    cublasDestroy(cublasHandle);
    return 0;
}

