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

float3 trilinearInterpolation(float3 pos, const float* lut, int lutSize, uint8_t bitDepth) {
    
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

	int xOffset = 1;
	int yOffset = lutSize;
	int zOffset = lutSize * lutSize;
    
    // multiply by the number of components in an RGB triplet (3)
	// c000
	int index0 = (intZ * zOffset + intY * yOffset + intX * xOffset);

	// c001
	int index1 = (index0 + zOffset);

	// c010
	int index2 = (index0 + yOffset);

	// c011
	int index3 = (index2 + zOffset);

	// c100
	int index4 = (index0 + xOffset);

	// c101
	int index5 = (index4 + zOffset);

	// c110
	int index6 = (index4 + yOffset);

	// c111
	int index7 = (index6 + zOffset);

	index0 *= 3;

	index1 *= 3;
	
	index2 *= 3;
	
	index3 *= 3;
	
	index4 *= 3;
	
	index5 *= 3;
	
	index6 *= 3;
	
	index7 *= 3;

    float3 c000 = make_float3(lut[index0], lut[index0 + 1], lut[index0 + 2]);
	float3 c001 = make_float3(lut[index1], lut[index1 + 1], lut[index1 + 2]);
	float3 c010 = make_float3(lut[index2], lut[index2 + 1], lut[index2 + 2]);
	float3 c011 = make_float3(lut[index3], lut[index3 + 1], lut[index3 + 2]);
	float3 c100 = make_float3(lut[index4], lut[index4 + 1], lut[index4 + 2]);
	float3 c101 = make_float3(lut[index5], lut[index5 + 1], lut[index5 + 2]);
	float3 c110 = make_float3(lut[index6], lut[index6 + 1], lut[index6 + 2]);
	float3 c111 = make_float3(lut[index7], lut[index7 + 1], lut[index7 + 2]);

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
void applyLUTKernel(const uint8_t* input, uint8_t* output, int rows, int cols, const uint8_t* lut) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < rows && j < cols) {
        int index = i * cols + j;
        int r = input[index * 3 + 2];
        int g = input[index * 3 + 1];
        int b = input[index * 3];

        // Apply the LUT to each pixel
        uint8_t R = lut[256 * 256 * 3 * r + 256 * 3 * g + 3 * b + 2];
        uint8_t G = lut[256 * 256 * 3 * r + 256 * 3 * g + 3 * b + 1];
        uint8_t B = lut[256 * 256 * 3 * r + 256 * 3 * g + 3 * b];

        // Store the result in the output vector
        output[index * 3 + 2] = R;
        output[index * 3 + 1] = G;
        output[index * 3] = B;
    }
}

// CUDA-accelerated function to apply LUT to the entire frame
cv::Mat applyLUTtoFrameCUDA(const uint8_t* frame, uint8_t* lut, int lutSize) {
    // Convert the frame to a vector of pixels
    int totalSize = H_BUFF * W_BUFF * 3;
    uint8_t* output = new uint8_t[totalSize];

    // Allocate GPU memory
    uint8_t* d_input;
    uint8_t* d_output;
    cudaMalloc(&d_input, totalSize * sizeof(uint8_t));
    cudaMalloc(&d_output, totalSize * sizeof(uint8_t));

    // Copy data to GPU
    cudaMemcpy(d_input, frame, totalSize * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, output, totalSize * sizeof(uint8_t), cudaMemcpyHostToDevice);

    // Calculate block and grid dimensions for the kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((H_BUFF + blockSize.x - 1) / blockSize.x, (W_BUFF + blockSize.y - 1) / blockSize.y);

    // Launch the kernel
    applyLUTKernel<<<gridSize, blockSize>>>(d_input, d_output, W_BUFF, H_BUFF, lut);

    cudaDeviceSynchronize();

    // check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "ERROR: %s \n", cudaGetErrorString(error));
    }

    // Copy the result back to CPU
    cudaMemcpy(output, d_output, totalSize * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_input);
    cudaFree(d_output);

    // Convert the vector of pixels back to a Mat
    cv::Mat result(H_BUFF, W_BUFF, CV_8UC3, output);
    delete[] output;
    return result;
}