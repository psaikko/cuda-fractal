#pragma once 

class FractalCompute {
public:
    FractalCompute(int buffer_w, int buffer_h);
    ~FractalCompute();

    void computeView(double r_min=-2, double r_max=1, double i_min=-1, double i_max=1);
    const double * getData();

private:

    double *gpu_data;
    int W;
    int H;
};
