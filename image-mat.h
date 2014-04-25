#pragma once
#include <cuda_runtime.h>

template <typename T> void image2MatGpu(const T*, const int, const int, const int, const int, const int, const int, const int, T*, cudaStream_t);

