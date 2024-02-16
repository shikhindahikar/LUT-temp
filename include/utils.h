#pragma once

#include <stdint.h>

__global__ void cudauyvy2bgr(int framesize, uint8_t *input, uint8_t *output);
__global__ void cudauyvy2rgb(int framesize, uint8_t *input, uint8_t *output);
