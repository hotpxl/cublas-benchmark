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
    void* devPtr[6];
    float alpha = 1.0, beta = 1.0;
    clock_t startTime;
    struct timeval tv1, tv2;
    assert(cublasCreate(&cublasHandle) == CUBLAS_STATUS_SUCCESS);
    assert(cudaMalloc(&devPtr[0], 96 * 363 * 4) == cudaSuccess);
    assert(cudaMalloc(&devPtr[1], 363 * 401408 * 4) == cudaSuccess);
    assert(cudaMalloc(&devPtr[2], 96 * 401408 * 4) == cudaSuccess);
    assert(cudaMalloc(&devPtr[3], 384 * 3456 * 4) == cudaSuccess);
    assert(cudaMalloc(&devPtr[4], 3456 * 21632 * 4) == cudaSuccess);
    assert(cudaMalloc(&devPtr[5], 384 * 21632 * 4) == cudaSuccess);
    cudaStream_t stream[16];
    for (int i = 0; i < 16; ++i) {
        assert(cudaStreamCreate(stream + i) == cudaSuccess);
    }
    int repeat = 30;
    int m1 = 96;
    int k1 = 363;
    int n1 = 401408;
    int i1 = repeat;
    int m2 = 384;
    int k2 = 3456;
    int n2 = 21632;
    int i2 = repeat;
    gettimeofday(&tv1, 0);
    while (i1--) {
        assert(cublasSetStream(cublasHandle, stream[i1 % 16]) == CUBLAS_STATUS_SUCCESS);
        assert(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n1, m1, k1, &alpha, (const float*) devPtr[1], n1, (const float*) devPtr[0], k1, &beta, (float*) devPtr[2], n1) == CUBLAS_STATUS_SUCCESS);
        assert(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n2, m2, k2, &alpha, (const float*) devPtr[4], n2, (const float*) devPtr[3], k2, &beta, (float*) devPtr[5], n2) == CUBLAS_STATUS_SUCCESS);
    }
    //
    // while (i2--) {
    //     assert(cublasSetStream(cublasHandle, stream[i2 % 16]) == CUBLAS_STATUS_SUCCESS);
    // }
    assert(cudaDeviceSynchronize() == cudaSuccess);
    gettimeofday(&tv2, 0);
    printf("%lf\n", (double) (tv2.tv_sec - tv1.tv_sec) * 1000000 + tv2.tv_usec - tv1.tv_usec);
    for (int i = 0; i < 16; ++i) {
        assert(cudaStreamDestroy(stream[i]) == cudaSuccess);
    }
    for (int i = 0; i < 6; ++i) {
        assert(cudaFree(devPtr[i]) == cudaSuccess);
    }
    cublasDestroy(cublasHandle);
    return 0;
}

