#pragma once 

class FractalCompute {
public:
    FractalCompute(int buffer_w, int buffer_h);
    ~FractalCompute();

    void computeView(float r_min=-2, float r_max=1, float i_min=-1, float i_max=1);
    const float * getData();

private:

    float *gpu_data;
    int W;
    int H;
};