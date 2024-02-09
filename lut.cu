#include "lut.h"

float3 Interpolate(float3 &v0, float3 &v1, float f)
{
	float3 out = make_float3(
		InterpolateHelper(v0.x, v1.x, f),
		InterpolateHelper(v0.y, v1.y, f),
		InterpolateHelper(v0.z, v1.z, f)
	);

	return out;
}

// interpolation: v0 + (v1 - v0) * f, v0/v1 = RGB component, f = weight
float InterpolateHelper(float v0, float v1, float f)
{
	return v0 + (v1 - v0) * f;
}

float3 trilinearInterpolation(float3 pos, float* lut, const size_t lutSize, uint8_t bitDepth) {
    
    const size_t totalLutSize = lutSize * lutSize * lutSize * 3;

    float R = static_cast<float>(pos.x);
    float G = static_cast<float>(pos.y);
    float B = static_cast<float>(pos.z);
    
    float normFactor = static_cast<float>((1 << bitDepth) - 1);

	R /= normFactor;
	G /= normFactor;
	B /= normFactor;
    
    // convert from point to grid coordinates
	float x = R * (lutSize - 1);
	float y = G * (lutSize - 1);
	float z = B * (lutSize - 1);

	// round down for coordinates
	int intX = static_cast<int>(x);
	int intY = static_cast<int>(y);
	int intZ = static_cast<int>(z);

    // difference between x and intX (floor of x)
    delta d;
    d.x = x - intX;
    d.y = y - intY;
    d.z = z - intZ;
    
    // calculate vertices in cube
	// cxyz is each coordinate, where xyz is a binary representation of a number
	// x changes fastest and represents the offset into a row, so add 1 each time it is set
	// y changes second fastest and represents a row in the cube. Add the lut size (the size of one dimension) when it is set
	// z changes the slowest and represents the plane in the cube. Add the square of the lut size when it is set

	size_t xOffset = 1 * 3;
	size_t yOffset = lutSize * 3;
	size_t zOffset = lutSize * lutSize * 3;

    // To prevent the out of bounds, we will start the index from 0 + difference again whenever an index is out of bounds
    
    // multiply by the number of components in an RGB triplet (3)
	// c000
	size_t index0 = (intZ * zOffset + intY * yOffset + intX * xOffset);

	// c001
	size_t index1 = (index0 + zOffset);
    if (index1 >= totalLutSize) {
        index1 -= totalLutSize;
    }

	// c010
	size_t index2 = (index0 + yOffset);
    if (index2 >= totalLutSize) {
        index2 -= totalLutSize;
    }

	// c011
	size_t index3 = (index2 + zOffset);
    if (index3 >= totalLutSize) {
        index3 -= totalLutSize;
    }

	// c100
	size_t index4 = (index0 + xOffset);
    if (index4 >= totalLutSize) {
        index4 -= totalLutSize;
    }

	// c101
	size_t index5 = (index4 + zOffset);
    if (index5 >= totalLutSize) {
        index5 -= totalLutSize;
    }

	// c110
	size_t index6 = (index4 + yOffset);
    if (index6 >= totalLutSize) {
        index6 -= totalLutSize;
    }

	// c111
	size_t index7 = (index6 + zOffset);
    if (index7 >= totalLutSize) {
        index7 -= totalLutSize;
    }


    float3 c000 = make_float3(static_cast<float>(lut[index0]), static_cast<float>(lut[index0 + 1]), static_cast<float>(lut[index0 + 2]));
	float3 c001 = make_float3(static_cast<float>(lut[index1]), static_cast<float>(lut[index1 + 1]), static_cast<float>(lut[index1 + 2]));
	float3 c010 = make_float3(static_cast<float>(lut[index2]), static_cast<float>(lut[index2 + 1]), static_cast<float>(lut[index2 + 2]));
	float3 c011 = make_float3(static_cast<float>(lut[index3]), static_cast<float>(lut[index3 + 1]), static_cast<float>(lut[index3 + 2]));
	float3 c100 = make_float3(static_cast<float>(lut[index4]), static_cast<float>(lut[index4 + 1]), static_cast<float>(lut[index4 + 2]));
	float3 c101 = make_float3(static_cast<float>(lut[index5]), static_cast<float>(lut[index5 + 1]), static_cast<float>(lut[index5 + 2]));
	float3 c110 = make_float3(static_cast<float>(lut[index6]), static_cast<float>(lut[index6 + 1]), static_cast<float>(lut[index6 + 2]));
	float3 c111 = make_float3(static_cast<float>(lut[index7]), static_cast<float>(lut[index7 + 1]), static_cast<float>(lut[index7 + 2]));

    // c00 -> interpolate c000 and c100
	float3 c00 = Interpolate(c000, c100, d.x);

	// c01 -> interpolate c001 and c101
	float3 c01 = Interpolate(c001, c101, d.x);

	// c10 -> interpolate c010 and c110
	float3 c10 = Interpolate(c010, c110, d.x);

	// c11 -> interpolate c011 and c111
	float3 c11 = Interpolate(c011, c111, d.x);

	// c0 -> interpolate c00 and c10
	float3 c0 = Interpolate(c00, c10, d.y);

	// c1 -> interpolate c01 and c11
	float3 c1 = Interpolate(c01, c11, d.y);

	// c -> interpolate c0 and c1
	float3 c = Interpolate(c0, c1, d.z);

    // create a vector out of all the coefficients
    float3 result = make_float3(static_cast<uint8_t>(c.x * normFactor), static_cast<uint8_t>(c.y * normFactor), static_cast<uint8_t>(c.z * normFactor));
    return result;
}

// Load the .cube LUT file
void getLutValues(std::string filename, int lutSize, float* values) {
    std::ifstream file(filename);
    std::cout << "Reading LUT file..." << std::endl;
    if (!file.is_open()) {
        std::cerr << "Error opening the file." << std::endl;
        exit(1);
    }
    
    // Read and ignore the title line
    std::string title;
    std::getline(file, title);
    title = title.substr(0, title.size() - 1); // Remove trailing newline

    // Read the LUT size
    std::string lutSizeLine;
    std::getline(file, lutSizeLine);
    std::istringstream lutSizeStream(lutSizeLine);
    std::string discard;
    int lut_size;

    lutSizeStream >> discard >> lut_size;
    
    if (lut_size != lutSize) {
        std::cerr << "Error: LUT size does not match the specified size." << std::endl;
        exit(1);
    }

    // Count the number of elements
    int elementCount = 0;
    std::string line;

    while (std::getline(file, line)) {
        std::istringstream valuesStream(line);
        std::string value;

        while (valuesStream >> value) {
            values[elementCount] = std::stof(value);
            elementCount++;
        }
    }

    file.close();
}

// CUDA kernel to apply LUT to each pixel in parallel
__global__
void applyLUTKernel(const uint8_t* input, uint8_t* output, int frameSize, const uint8_t* lut) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    frameSize >>= 1;
    for(int i = index; i < frameSize; i += stride) {
        // UYV values from UYVY frame
        uint8_t U = input[(i << 2)];
        uint8_t Y1 = input[(i << 2) + 1];
        uint8_t V = input[(i << 2) + 2];
        uint8_t Y2 = input[(i << 2) + 3];

        uint8_t pixel1U = lut[256 * 256 * 3 * U + 256 * 3 * Y1 + 3 * V];
        uint8_t pixel1Y = lut[256 * 256 * 3 * U + 256 * 3 * Y1 + 3 * V + 1];
        uint8_t pixel1V = lut[256 * 256 * 3 * U + 256 * 3 * Y1 + 3 * V + 2];

        uint8_t pixel2U = lut[256 * 256 * 3 * U + 256 * 3 * Y2 + 3 * V];
        uint8_t pixel2Y = lut[256 * 256 * 3 * U + 256 * 3 * Y2 + 3 * V + 1];
        uint8_t pixel2V = lut[256 * 256 * 3 * U + 256 * 3 * Y2 + 3 * V + 2];

        // if (pixel1Y != pixel2Y) {
        //     printf("------------------------------------------------------------------------------------\n");
        //     printf("pixel1U: %d, pixel1V: %d, pixel2U: %d, pixel2V: %d\n", pixel1U, pixel1V, pixel2U, pixel2V);
        //     printf("------------------------------------------------------------------------------------\n");
        // }

        // getting corresponding LUT[U1][Y1][V1] values to put back into the frame
        output[(i << 2)] = (pixel1U + pixel2U) >> 1;
        output[(i << 2) + 1] = pixel1Y; 
        output[(i << 2) + 2] = (pixel1V + pixel2V) >> 1;
        output[(i << 2) + 3] = pixel2Y;

    }
}

// CUDA-accelerated function to apply LUT to the entire frame
uint8_t* applyLUTtoFrameCUDA(const uint8_t* frame, uint8_t* lut) {
    // Convert the frame to a vector of pixels
    int totalSize = H_BUFF * W_BUFF * 2;
    uint8_t* output = new uint8_t[totalSize];

    // Allocate GPU memory
    uint8_t* d_output;
    cudaMalloc(&d_output, totalSize * sizeof(uint8_t));

    // Copy data to GPU
    cudaMemcpy(d_output, output, totalSize * sizeof(uint8_t), cudaMemcpyHostToDevice);

    // Launch the kernel to apply UYVY LUT to each pixel in UYVY frame
    applyLUTKernel<<<960, 256>>>(frame, d_output, W_BUFF * H_BUFF, lut);

    cudaDeviceSynchronize();

    // check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "ERROR @ applying LUT: %s \n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    // Copy the result back to CPU
    cudaMemcpy(output, d_output, totalSize * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_output);

    return output;
}