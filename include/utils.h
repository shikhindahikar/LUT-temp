#pragma once

#include <stdint.h>

__global__ void cudauyvy2rgb(int framesize, uint8_t *input, uint8_t *output);
__global__ void cudargblut2uyvy(int totalLutSize, uint8_t *interpVals, uint8_t *uyvyLut);
