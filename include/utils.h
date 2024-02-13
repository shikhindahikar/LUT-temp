#pragma once

#include <stdint.h>

__global__ void cudauyvy2bgr(int framesize, uint8_t *input, uint8_t *output);
__global__ void cudargblut2yuv(uint8_t *input, uint8_t *output, uint8_t* yuv2rgbIdxLut);
__global__ void cudargbIdx2yuv(uint8_t* output);
