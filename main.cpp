#include "image-mat.h"
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

struct ConvParameter {
    int kernelLength;
    int channelNum;
    int stride;
    int padding;
    int height;
    int width;
};

int main() {
    cublasHandle_t cublasHandle;
    void* devPtr[8];
    float alpha = 1.0, beta = 1.0;
    clock_t startTime;
    struct timeval tv1, tv2;
    int repeat = 30;
    int num = 128;
    int height1 = 227;
    int width1 = 227;
    int channelNum1 = 3;
    int kernelLength1 = 11;
    int stride1 = 4;
    int padding1 = 2;
    int outputHeight1 = (height1 + 2 * padding1 - kernelLength1) / stride1 + 1;
    int outputWidth1 = (width1 + 2 * padding1 - kernelLength1) / stride1 + 1;
    int m1 = 96;
    int k1 = channelNum1 * kernelLength1 * kernelLength1;
    int n1 = outputHeight1 * outputWidth1 * num;
    int i1 = repeat;
    int height2 = 13;
    int width2 = 13;
    int channelNum2 = 384;
    int kernelLength2 = 3;
    int stride2 = 1;
    int padding2 = 1;
    int outputHeight2 = (height2 + 2 * padding2 - kernelLength2) / stride2 + 1;
    int outputWidth2 = (width2 + 2 * padding2 - kernelLength2) / stride2 + 1;
    int m2 = 384;
    int k2 = channelNum2 * kernelLength2 * kernelLength2;
    int n2 = outputHeight2 * outputWidth2 * num;
    int i2 = repeat;
    assert(cublasCreate(&cublasHandle) == CUBLAS_STATUS_SUCCESS);
    assert(cudaMalloc(&devPtr[0], m1 * k1 * sizeof(float)) == cudaSuccess);
    assert(cudaMalloc(&devPtr[1], k1 * n1 * sizeof(float)) == cudaSuccess);
    assert(cudaMalloc(&devPtr[2], m1 * n1 * sizeof(float)) == cudaSuccess);
    assert(cudaMalloc(&devPtr[3], m2 * k2 * sizeof(float)) == cudaSuccess);
    assert(cudaMalloc(&devPtr[4], k2 * n2 * sizeof(float)) == cudaSuccess);
    assert(cudaMalloc(&devPtr[5], m2 * n2 * sizeof(float)) == cudaSuccess);
    assert(cudaMalloc(&devPtr[6], height1 * width1 * channelNum1 * num * sizeof(float)) == cudaSuccess);
    assert(cudaMalloc(&devPtr[7], height2 * width2 * channelNum2 * num * sizeof(float)) == cudaSuccess);
    cudaStream_t stream[16];
    for (int i = 0; i < 16; ++i) {
        assert(cudaStreamCreate(stream + i) == cudaSuccess);
    }
    gettimeofday(&tv1, 0);
    while (i1--) {
        assert(cublasSetStream(cublasHandle, stream[i1 % 16]) == CUBLAS_STATUS_SUCCESS);
        image2MatGpu((const float*) devPtr[6], num, channelNum1, height1, width1, kernelLength1, padding1, stride1, (float*) devPtr[1], stream[i1 % 16]);
        assert(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n1, m1, k1, &alpha, (const float*) devPtr[1], n1, (const float*) devPtr[0], k1, &beta, (float*) devPtr[2], n1) == CUBLAS_STATUS_SUCCESS);
        assert(cublasSetStream(cublasHandle, stream[(i1 + 1) % 16]) == CUBLAS_STATUS_SUCCESS);
        image2MatGpu((const float*) devPtr[7], num, channelNum2, height2, width2, kernelLength2, padding2, stride2, (float*) devPtr[4], stream[(i1 + 1) % 16]);
        assert(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n2, m2, k2, &alpha, (const float*) devPtr[4], n2, (const float*) devPtr[3], k2, &beta, (float*) devPtr[5], n2) == CUBLAS_STATUS_SUCCESS);
    }
    //
    assert(cudaDeviceSynchronize() == cudaSuccess);
    gettimeofday(&tv2, 0);
    printf("%lf\n", (double) (tv2.tv_sec - tv1.tv_sec) * 1000000 + tv2.tv_usec - tv1.tv_usec);
    for (int i = 0; i < 16; ++i) {
        assert(cudaStreamDestroy(stream[i]) == cudaSuccess);
    }
    for (int i = 0; i < 8; ++i) {
        assert(cudaFree(devPtr[i]) == cudaSuccess);
    }
    cublasDestroy(cublasHandle);
    return 0;
}

