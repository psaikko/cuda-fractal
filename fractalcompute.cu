#include "fractalcompute.h"

__device__
double2 eval_fc(double2 z, double2 c) {
    // compute f_c(z) = z^2 + c 
    // for complex z, c
    // where real(z) = z.x, im(z) = z.y

    double res_r = z.x*z.x - z.y*z.y + c.x;
    double res_i = 2*z.x*z.y + c.y;

    return make_double2(res_r, res_i);
}

__global__ 
void cuda_iterate(int n_iters, double threshold, 
                  int W, int H, double* data,
                  double r_min, double r_max, 
                  double i_min, double i_max) 
{
    // Compute index and stride in the usual way
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int k = index; k < W*H; k += stride) {
        // Compute pixel x,y location
        int x = k % W;
        int y = k / W;

        // Compute representative point on the view of the complex plane
        double r = r_min + (r_max - r_min) / double(W) * double(x);
        double i = i_min + (i_max - i_min) / double(H) * double(y);

        // Initialize z and c
        double2 c = make_double2(r, i);
        double2 z = make_double2(0, 0);

        // Iterate f_c on z
        int j = 0;
        for (; j < n_iters; ++j) {
            z = eval_fc(z, c);
            // abs(z) > threshold?
            if (z.x * z.x + z.y * z.y > threshold * threshold) 
                break;
            j++;
        }

        data[k] = double(j) / double(n_iters);
    }
}

FractalCompute::FractalCompute(int buffer_w, int buffer_h) :
    W(buffer_w), 
    H(buffer_h) 
{
    cudaMallocManaged(&gpu_data, sizeof(double) * W * H);
}

FractalCompute::~FractalCompute() {
    cudaFree(gpu_data);
}

#define N_ITERS 1000
#define THRESHOLD 2

void FractalCompute::computeView(double r_min, double r_max, double i_min, double i_max) {
    int blockSize = 128;
    int nBlocks = (W*H + blockSize - 1) / blockSize;

    cuda_iterate<<<nBlocks, blockSize>>>(N_ITERS, THRESHOLD, W, H, gpu_data, r_min, r_max, i_min, i_max);
}

const double * FractalCompute::getData() {
    cudaDeviceSynchronize();
    return gpu_data;
}