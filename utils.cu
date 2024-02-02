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
