#pragma once 

class FractalCompute {
public:
    FractalCompute(int buffer_w, int buffer_h);
    ~FractalCompute();

    void computeView(double r_min=-2, double r_max=1, double i_min=-1, double i_max=1, float hue_offset=0);
    const unsigned char * getData();

private:

    float *gpu_data;
    unsigned char *hsv_buffer;
    unsigned char *rgb_buffer;

    int W;
    int H;
};
