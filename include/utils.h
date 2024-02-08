#pragma once

#include <stdint.h>

__global__ void cudauyvy2bgr(int framesize, uint8_t *input, uint8_t *output);
__global__ void cudargblut2yuv(int framesize, uint8_t *input, uint8_t *output);
__global__ void cudayuv2uyvy(int framesize, uint8_t *input, uint8_t *output);
__global__ void cudauyvy2yuv(int framesize, uint8_t *input, uint8_t *output);
