#include "fractalcompute.h"

__device__
float2 eval_fc(float2 z, float2 c) {
    // compute f_c(z) = z^2 + c 
    // for complex z, c
    // where real(z) = z.x, im(z) = z.y

    float res_r = z.x*z.x - z.y*z.y + c.x;
    float res_i = 2*z.x*z.y + c.y;

    return make_float2(res_r, res_i);
}

__global__ 
void cuda_iterate(int n_iters, float threshold, 
                  int W, int H, float* data,
                  float r_min, float r_max, 
                  float i_min, float i_max) 
{
    // Compute index and stride in the usual way
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int k = index; k < W*H; k += stride) {
        // Compute pixel x,y location
        int x = k % W;
        int y = k / W;

        // Compute representative point on the view of the complex plane
        float r = r_min + (r_max - r_min) / float(W) * float(x);
        float i = i_min + (i_max - i_min) / float(H) * float(y);

        // Initialize z and c
        float2 c = make_float2(r, i);
        float2 z = make_float2(0, 0);

        // Iterate f_c on z
        int j = 0;
        for (; j < n_iters; ++j) {
            z = eval_fc(z, c);
            // abs(z) > threshold?
            if (z.x * z.x + z.y * z.y > threshold * threshold) 
                break;
            j++;
        }

        data[k] = float(j) / float(n_iters);
    }
}

FractalCompute::FractalCompute(int buffer_w, int buffer_h) :
    W(buffer_w), 
    H(buffer_h) 
{
    cudaMallocManaged(&gpu_data, sizeof(int) * W * H);
}

FractalCompute::~FractalCompute() {
    cudaFree(gpu_data);
}

#define N_ITERS 1000
#define THRESHOLD 2

void FractalCompute::computeView(float r_min, float r_max, float i_min, float i_max) {
    int blockSize = 128;
    int nBlocks = (W*H + blockSize - 1) / blockSize;

    cuda_iterate<<<nBlocks, blockSize>>>(N_ITERS, THRESHOLD, W, H, gpu_data, r_min, r_max, i_min, i_max);
}

const float * FractalCompute::getData() {
    cudaDeviceSynchronize();
    return gpu_data;
}