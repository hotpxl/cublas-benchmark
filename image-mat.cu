#include "image-mat.h"
#include <cmath>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

using namespace std;

#define CAFFE_CUDA_NUM_THREADS (1024)
#define CAFFE_GET_BLOCKS(n) (((n) + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS)

template <typename T> __global__ void image2MatGpuKernel(const int n, const T* dataImage, const int inputNum, const int height, const int width, const int kernelLength, const int channelNum, const int pad, const int stride, const int nextLayerHeight, const int nextLayerWidth, T* dataCol) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < n; index += blockDim.x * gridDim.x) {
    int widthOffsetInOutput = index % nextLayerWidth;
    int heightOffsetInOutput = (index / nextLayerWidth) % nextLayerHeight;
    int imageIdx = index / nextLayerWidth / nextLayerHeight % inputNum;
    int filterIdx = index / nextLayerWidth / nextLayerHeight / inputNum;
    int filterOffsetInOutput = filterIdx * kernelLength * kernelLength;
    int heightOffsetInInput = heightOffsetInOutput * stride - pad;
    int widthOffsetInInput = widthOffsetInOutput * stride - pad;
    dataCol += ((filterOffsetInOutput * inputNum + imageIdx) * nextLayerHeight + heightOffsetInOutput) * nextLayerWidth + widthOffsetInOutput;
    dataImage += ((imageIdx * channelNum + filterIdx) * height + heightOffsetInInput) * width + widthOffsetInInput;
    for (int i = 0; i < kernelLength; ++i) {
      for (int j = 0; j < kernelLength; ++j) {
        int h = heightOffsetInInput + i;
        int w = widthOffsetInInput + j;
        *dataCol = (0 <= h && 0 <= w && w < width && h < height) ? dataImage[i * width + j] : 0;
        dataCol += nextLayerHeight * nextLayerWidth * inputNum;
      }
    }
  }
}

template <typename T> void image2MatGpu(const T* dataImage, const int inputNum, const int channelNum, const int height, const int width, const int kernelLength, const int pad, const int stride, T* dataCol, cudaStream_t stream) {
  // We are going to launch channelNum * nextLayerHeight * nextLayerWidth kernels, each kernel responsible for copying a single-channel grid.
  int nextLayerHeight = (height + 2 * pad - kernelLength) / stride + 1;
  int nextLayerWidth = (width + 2 * pad - kernelLength) / stride + 1;
  int kernelNum = inputNum * channelNum * nextLayerHeight * nextLayerWidth;
  image2MatGpuKernel<T><<<CAFFE_GET_BLOCKS(kernelNum), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(kernelNum, dataImage, inputNum, height, width, kernelLength, channelNum, pad, stride, nextLayerHeight, nextLayerWidth, dataCol);
  if (cudaGetLastError() != cudaSuccess) {
    throw runtime_error("Device kernel failed");
  }
}

template void image2MatGpu<float>(const float*, const int, const int, const int, const int, const int, const int, const int, float*, cudaStream_t);

