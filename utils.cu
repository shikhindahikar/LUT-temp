#include "utils.h"

__global__ void cudauyvy2rgb(int framesize, uint8_t *input, uint8_t *output) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	framesize >>= 1;
	for(int i = index; i < framesize; i += stride) {
		float u = input[(i << 2)];
		float y = input[(i << 2) + 1];
		float v = input[(i << 2) + 2];
		y -= 16;
		u -= 128;
		v -= 128;
		float red = y * 1.164 + v * 1.596;
		float green = y * 1.164 - u * 0.392 - v * 0.813;
		float blue = y * 1.164 + u * 2.017;
		if(red > 255) red = 255;
		else if(red < 0) red = 0;
		if(green > 255) green = 255;
		else if(green < 0) green = 0;
		if(blue > 255) blue = 255;
		else if(blue < 0) blue = 0;
		output[i * 6] = red;
		output[i * 6 + 1] = green;
		output[i * 6 + 2] = blue;

		y = input[(i << 2) + 3];
		y -= 16;
		red = y * 1.164 + v * 1.596;
		green = y * 1.164 - u * 0.392 - v * 0.813;
		blue = y * 1.164 + u * 2.017;
		if(red > 255) red = 255;
		else if(red < 0) red = 0;
		if(green > 255) green = 255;
		else if(green < 0) green = 0;
		if(blue > 255) blue = 255;
		else if(blue < 0) blue = 0;
		output[i * 6 + 3] = red;
		output[i * 6 + 4] = green;
		output[i * 6 + 5] = blue;
	}
}

__global__ void cudargblut2uyvy(int totalLutSize, uint8_t *interpVals, uint8_t *uyvyLut) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	totalLutSize >>= 1;
	for(int i = index; i < totalLutSize; i += stride) {
		// For first pixel in the pair
		float red = interpVals[i * 6];
		float green = interpVals[i * 6 + 1];
		float blue = interpVals[i * 6 + 2];
		float y = 16 + 0.256 * red // Red
			+ 0.504 * green // Green
			+ 0.0979 * blue; // Blue
		if(y > 255) y = 255;
		float v = 128 - 0.148 * red  - 0.291 * green + 0.439 * blue;
		if(v < 0) v = 0;
		else if(v > 255) v = 255;
		float u = 128 + 0.439 * red - 0.368 * green - 0.0714 * blue;
		if(u < 0) u = 0;
		else if(u > 255) u = 255;
		uyvyLut[(i << 2)] = u;
		uyvyLut[(i << 2) + 1] = y;
		uyvyLut[(i << 2) + 2] = v;
		// For second pixel in the pair
		red = interpVals[i * 6 + 3];
		green = interpVals[i * 6 + 4];
		blue = interpVals[i * 6 + 5];
		y = 16 + 0.256 * red // Red
			+ 0.504 * green // Green
			+ 0.0979 * blue; // Blue
		if(y > 255) y = 255;
		uyvyLut[(i << 2) + 3] = y;
	}
}