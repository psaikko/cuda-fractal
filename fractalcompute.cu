#include "fractalcompute.h"
#include <npp.h>

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
void iterate_points(int n_iters, double threshold, 
                    int W, int H, float* data,
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

        data[k] = float(j) / float(n_iters);
    }
}

__global__
void depth_to_hsv(int W, int H, float* depth_data, unsigned char *hsv_buffer, float hue_offset) {
    // Compute index and stride in the usual way
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int k = index; k < W*H; k += stride) { 
        unsigned char h = int(depth_data[k] * 255.0 + hue_offset) % 255; 
        unsigned char v = abs(depth_data[k] - 1.0) > 1e-4 ? 255 : 0;

        hsv_buffer[k*4 + 0] = h;   // H
        hsv_buffer[k*4 + 1] = 255; // S [0: white, 255: saturated]
        hsv_buffer[k*4 + 2] = v;   // V [0: black, 255: color]
        hsv_buffer[k*4 + 3] = 255; // A
    }
}

FractalCompute::FractalCompute(int buffer_w, int buffer_h) :
    W(buffer_w), 
    H(buffer_h) 
{
    cudaMallocManaged(&gpu_data, sizeof(float) * W * H);
    cudaMallocManaged(&hsv_buffer, sizeof(unsigned char) * 4 * W * H);
    cudaMallocManaged(&rgb_buffer, sizeof(unsigned char) * 4 * W * H);
    memset(rgb_buffer, 255, 4*W*H);
}

FractalCompute::~FractalCompute() {
    cudaFree(gpu_data);
    cudaFree(hsv_buffer);
    cudaFree(rgb_buffer);
}

#define N_ITERS 600
#define THRESHOLD 2

void FractalCompute::computeView(double r_min, double r_max, double i_min, double i_max, float hue_offset) {
    int blockSize = 128;
    int nBlocks = (W*H + blockSize - 1) / blockSize;

    iterate_points<<<nBlocks, blockSize>>>(N_ITERS, THRESHOLD, W, H, gpu_data, r_min, r_max, i_min, i_max);
    depth_to_hsv<<<nBlocks, blockSize>>>(W, H, gpu_data, hsv_buffer, hue_offset);
    nppiHSVToRGB_8u_AC4R(hsv_buffer, 4 * W, rgb_buffer, 4 * W, {W, H});
}

const unsigned char * FractalCompute::getData() {
    cudaDeviceSynchronize();
    return rgb_buffer;
}