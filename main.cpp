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

int main() {
    void* devPtr, * hostPtr;
    int kCandidate[] = {363, 2304, 2400, 3456};
    clock_t startTime;
    struct timeval tv1, tv2;
    assert(cudaMalloc(&devPtr, (1 << 30)) == cudaSuccess);
    assert(hostPtr = malloc(1 << 30));

    for (int kIdx = 0; kIdx < 4; ++kIdx) {
        int k = kCandidate[kIdx];
        int nMax = (1 << 28) / k;
        int nMaxRounded;
        asm("bsrl %1, %0"
                : "=r" (nMaxRounded)
                : "r" (nMax));
        for (int n = 512; n <= (1 << nMaxRounded); n <<= 1) {
            int repeat = ((double) ((unsigned long long) 1 << 39)) / k / n;
            int i = repeat;
            int channelNum = 3;
            gettimeofday(&tv1, 0);
            while (i--) {
                image2MatGpu(devPtr,  channelNum, 
                assert(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, (const float*) devPtr, n, (const float*) devPtr, k, &beta, (float*) devPtr, n) == CUBLAS_STATUS_SUCCESS);
            }
            assert(cudaDeviceSynchronize() == cudaSuccess);
            gettimeofday(&tv2, 0);
            printf("%d %d %d %d %lf\n", m, k, n, repeat, (double) (tv2.tv_sec - tv1.tv_sec) * 1000000 + tv2.tv_usec - tv1.tv_usec);
        }
    }
    free(hostPtr);
    assert(cudaFree(devPtr) == cudaSuccess);
    return 0;
}

