#include "utils.h"

__global__ void cudauyvy2bgr(int framesize, uint8_t *input, uint8_t *output) {
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
		output[i * 6] = blue;
		output[i * 6 + 1] = green;
		output[i * 6 + 2] = red;

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
		output[i * 6 + 3] = blue;
		output[i * 6 + 4] = green;
		output[i * 6 + 5] = red;
	}
}

__global__ void cudargblut2yuv(uint8_t *input, uint8_t *output) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int idz = blockIdx.z * blockDim.z + threadIdx.z;
	if (idx < 256 && idy < 256 && idz < 256) {
		int index = idx * 256 * 256 * 3 + idy * 256 * 3 + idz * 3;
		float red = input[index];
		float green = input[index + 1];
		float blue = input[index + 2];
		float y = 16 + 0.256 * red + 0.504 * green + 0.0979 * blue;
		float u = 128 + 0.439 * red - 0.368 * green - 0.0714 * blue;
		float v = 128 - 0.148 * red  - 0.291 * green + 0.439 * blue;
		
		// clamping 0-255
		if(y > 255) y = 255;
		else if(y < 0) y = 0;
		if(u < 0) u = 0;
		else if(u > 255) u = 255;
		if(v < 0) v = 0;
		else if(v > 255) v = 255;
		
		output[index] = u;
		output[index + 1] = y;
		output[index + 2] = v;
	}
}
