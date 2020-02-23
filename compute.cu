#include "compute.h"
#include "cuda.h"
#include "cuda_runtime.h"

__global__
void cuda_init(int N, float *p, float v) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N; i += stride) {
        p[i] = v;
    }
}

__global__
void cuda_add(int N, float *a, float *b) { 
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N; i += stride) {
        b[i] += a[i];
    }
}

float compute() {
    int N = 1000000;
    float *a, *b;

    cudaMallocManaged(&a, sizeof(float) * N);
    cudaMallocManaged(&b, sizeof(float) * N);

    int blockSize = 128;
    int nBlocks = (N + blockSize - 1) / blockSize;

    cuda_init<<<nBlocks, blockSize>>>(N, a, 1.0);
    cuda_init<<<nBlocks, blockSize>>>(N, b, 2.0);
    cuda_add<<<nBlocks, blockSize>>>(N, a, b);

    cudaDeviceSynchronize();

    float err = 0.0;

    for (int i = 0; i < N; ++i) 
        err += (b[i] - 3.0);

    return err;
}