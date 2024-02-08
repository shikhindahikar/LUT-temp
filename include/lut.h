#include <iostream>
#include <fstream>
#include <sstream>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include "NvInfer.h"

#define H_BUFF 1080
#define W_BUFF 1920

struct delta {
    float x;
    float y;
    float z;
};

float3 trilinearInterpolation(float3 pos, float* lut, const size_t lutSize, uint8_t bitDepth);

float InterpolateHelper(float v0, float v1, float t);

float3 Interpolate(float3 &v0, float3 &v1, float f);

void getLutValues(std::string filename, int lutSize, float* values);

__global__ void applyLUTKernel(const uint8_t* input, uint8_t* output, int frameSize, const uint8_t* lut);

uint8_t* applyLUTtoFrameCUDA(const uint8_t* frame, uint8_t* lut);