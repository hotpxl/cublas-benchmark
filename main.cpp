#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cassert>
#include <cstdlib>
#include <ctime>
#include <cstdio>

using namespace std;

#define MAX_LENGTH (1 << 14)

int main() {
    cublasHandle_t cublasHandle;
    void* devPtr[3], * hostPtr[3];
    int m, n, k;
    float alpha, beta;
    clock_t startTime;

    assert(cublasCreate(&cublasHandle) == CUBLAS_STATUS_SUCCESS);
    m = n = k = MAX_LENGTH;
    alpha = beta = 1.0;
    assert(cudaMalloc(&devPtr[0], m * k * sizeof(float)) == cudaSuccess);
    assert(cudaMalloc(&devPtr[1], k * n * sizeof(float)) == cudaSuccess);
    assert(cudaMalloc(&devPtr[2], m * n * sizeof(float)) == cudaSuccess);
    assert(hostPtr[0] = malloc(m * k * sizeof(float)));
    assert(hostPtr[1] = malloc(k * n * sizeof(float)));
    assert(hostPtr[2] = malloc(m * n * sizeof(float)));

    // for (int size = 1024; size <= MAX_LENGTH; size <<= 1) {
        // int repeat = (MAX_LENGTH / size);
        int repeat = 1024;
        // m = n = k = MAX_LENGTH;
        m = n = k = 4096;
        startTime = clock();
        while (repeat--) {
            assert(cublasSetMatrix(m, k, sizeof(float), hostPtr[0], k, devPtr[0], k) == CUBLAS_STATUS_SUCCESS);
            assert(cublasSetMatrix(k, n, sizeof(float), hostPtr[1], n, devPtr[1], n) == CUBLAS_STATUS_SUCCESS);
            assert(cublasSetMatrix(m, n, sizeof(float), hostPtr[2], n, devPtr[0], n) == CUBLAS_STATUS_SUCCESS);
            assert(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, (const float*) devPtr[1], n, (const float*) devPtr[0], k, &beta, (float*) devPtr[2], n) == CUBLAS_STATUS_SUCCESS);
            assert(cublasGetMatrix(m, k, sizeof(float), devPtr[0], k, hostPtr[0], k) == CUBLAS_STATUS_SUCCESS);
            assert(cublasGetMatrix(k, n, sizeof(float), devPtr[1], n, hostPtr[1], n) == CUBLAS_STATUS_SUCCESS);
            assert(cublasGetMatrix(m, n, sizeof(float), devPtr[2], n, hostPtr[0], n) == CUBLAS_STATUS_SUCCESS);
        }
        // printf("Size: %d, time: %lf\n", size, (double) (clock() - startTime) / size / size / size);
    // }

    free(hostPtr[2]);
    free(hostPtr[1]);
    free(hostPtr[0]);
    assert(cudaFree(devPtr[2]) == cudaSuccess);
    assert(cudaFree(devPtr[1]) == cudaSuccess);
    assert(cudaFree(devPtr[0]) == cudaSuccess);
    cublasDestroy(cublasHandle);
    return 0;
}

