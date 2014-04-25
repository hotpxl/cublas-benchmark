#include "image-mat.h"
#include <cuda_runtime.h>
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
    void* devPtr, * hostPtr;
    clock_t startTime;
    struct timeval tv1, tv2;
    struct ConvParameter params[4];
    cudaStream_t stream[16];
    assert(cudaMalloc(&devPtr, (1 << 30)) == cudaSuccess);
    assert(hostPtr = malloc(1 << 30));
    params[0].kernelLength = 11;
    params[0].channelNum = 3;
    params[0].stride = 4;
    params[0].padding = 2;
    params[0].height = 227;
    params[0].width = 227;
    params[1].kernelLength = 3;
    params[1].channelNum = 256;
    params[1].stride = 1;
    params[1].padding = 1;
    params[1].height = 13;
    params[1].width = 13;
    params[2].kernelLength = 5;
    params[2].channelNum = 96;
    params[2].stride = 1;
    params[2].padding = 1;
    params[2].height = 28;
    params[2].width = 28;
    params[3].kernelLength = 3;
    params[3].channelNum = 384;
    params[3].stride = 1;
    params[3].padding = 1;
    params[3].height = 13;
    params[3].width = 13;
    for (int i = 0; i < 16; ++i) {
        cudaStreamCreate(&stream[i]);
    }
    for (int paramIdx = 0; paramIdx < 4; ++paramIdx) {
        struct ConvParameter param = params[paramIdx];
        int outputHeight = (param.height + 2 * param.padding - param.kernelLength) / param.stride + 1;
        int outputWidth = (param.width + 2 * param.padding - param.kernelLength) / param.stride + 1;
        int concatMax = (double) (1 << 28) / param.channelNum / param.kernelLength / param.kernelLength / outputHeight / outputWidth;
        for (int n = 1; n <= concatMax; n = 1.2 * n + 1) {
            int repeat = ((double) ((unsigned long long) 1 << 36)) / param.channelNum / param.kernelLength / param.kernelLength / outputHeight / outputWidth / n;
            int i = repeat;
            gettimeofday(&tv1, 0);
            while (i--) {
                image2MatGpu((const float*) devPtr, n, param.channelNum, param.height, param.width, param.kernelLength, param.padding, param.stride, (float*) devPtr, stream[i % 16]);
            }
            assert(cudaDeviceSynchronize() == cudaSuccess);
            gettimeofday(&tv2, 0);
            printf("k = %d each = %d total = %d repeat = %d time = %lf\n", param.channelNum * param.kernelLength * param.kernelLength, outputHeight * outputWidth, n, repeat, (double) (tv2.tv_sec - tv1.tv_sec) * 1000000 + tv2.tv_usec - tv1.tv_usec);
        }
    }

    for (int i = 0; i < 16; ++i) {
        cudaStreamDestroy(stream[i]);
    }
    free(hostPtr);
    assert(cudaFree(devPtr) == cudaSuccess);
    return 0;
}

