#pragma once

#include <stdint.h>

__global__ void cudauyvy2rgb(int framesize, uint8_t *input, uint8_t *output);
