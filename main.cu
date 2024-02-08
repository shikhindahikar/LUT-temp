#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "lut.h"
#include <cstdlib>
#include "utils.h"
#include <memory>

int main(int argc, char* argv[]) {
    // Specify the input video file and the output directory
    std::string inputVideoFile = "/media/ozsports/ozsports/Work/ball_far.mp4";

    // Specify the size of the LUT and path to the .cube file
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <LUT size> <LUT file path>" << std::endl;
        exit(EXIT_FAILURE);
    }

    const size_t LUT_SIZE = std::stoi(argv[1]);
    std::cout << "LUT size: " << LUT_SIZE << std::endl;
    std::string lutFile(argv[2]);

    std::cout << "LUT file path: " << lutFile << std::endl;

    // Read the LUT
    float* lutValues = new float[LUT_SIZE * LUT_SIZE * LUT_SIZE * 3];
    getLutValues(lutFile, LUT_SIZE, lutValues);

    // interpolated LUT values for all 256^3 RGB values
    uint8_t* interpolatedLUTValues = new uint8_t[256 * 256 * 256 * 3];

    std::cout << "Interpolating LUT values..." << std::endl;

    float3 rgb;
    const uint8_t bitDepth = 8;
    // interpolate the LUT values for all 256^3 RGB values
    for (int i = 0; i < 256; i++) {
        for (int j = 0; j < 256; j++){
            for (int k = 0; k < 256; k++) {
                rgb = trilinearInterpolation(make_float3(i, j, k), lutValues, LUT_SIZE, bitDepth);
                interpolatedLUTValues[i * 256 * 256 * 3 + j * 256 * 3 + k * 3] = static_cast<uint8_t>(rgb.x);
                interpolatedLUTValues[i * 256 * 256 * 3 + j * 256 * 3 + k * 3 + 1] = static_cast<uint8_t>(rgb.y);
                interpolatedLUTValues[i * 256 * 256 * 3 + j * 256 * 3 + k * 3 + 2] = static_cast<uint8_t>(rgb.z);
            }
        }
    }

    // free memory
    delete[] lutValues;

    
    uint8_t* d_interpolatedLUTValues;
    cudaMalloc((void**)&d_interpolatedLUTValues, 256 * 256 * 256 * 3 * sizeof(uint8_t));
    cudaMemcpy(d_interpolatedLUTValues, interpolatedLUTValues, 256 * 256 * 256 * 3 * sizeof(uint8_t), cudaMemcpyHostToDevice);
    delete[] interpolatedLUTValues;

    // allocate memory for the LUT values converted to UYVY
    uint8_t* d_lutUYVY;
    cudaMalloc((void**)&d_lutUYVY, 256 * 256 * 256 * 3 * sizeof(uint8_t));

    // converting the interpolated RGB LUT to UYVY LUT on CUDA
    cudargb2yuv<<<960, 256>>>(256 * 256 * 256, d_interpolatedLUTValues, d_lutUYVY);

    // check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "ERROR @ RGB LUT 2 UYVY: %s \n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    cudaFree(d_interpolatedLUTValues);

    std::string ffmpeg_path = "ffmpeg";

    // command to run ffmpeg
	std::string command = ffmpeg_path + " -i " + inputVideoFile + " -f image2pipe -pix_fmt uyvy422 -vcodec rawvideo -";
	std::FILE* pipe = popen(command.c_str(), "r");
	if (!pipe) {
		throw std::runtime_error("Failed to open pipe");
	}

	void* raw_image = malloc(H_BUFF * W_BUFF * 2 * sizeof(uint8_t));
	// Run inference on each frame till EOF
	while (fread(raw_image, H_BUFF * W_BUFF * 2, 1, pipe) == 1) {
        // put raw_image into uint8_t* frame
        uint8_t* frame = new uint8_t[H_BUFF * W_BUFF * 3];
        uint8_t* raw_video_data = (uint8_t*)raw_image;
        
        // put frame into device memory
        uint8_t* d_frame;
        cudaMalloc((void**)&d_frame, H_BUFF * W_BUFF * 2 * sizeof(uint8_t));
        cudaMemcpy(d_frame, raw_video_data, H_BUFF * W_BUFF * 2 * sizeof(uint8_t), cudaMemcpyHostToDevice);

        // apply LUTs
        uint8_t* final_frame = applyLUTtoFrameCUDA(d_frame, d_lutUYVY);
        cudaFree(d_frame);

        // put final frame into device memory for converting UYVY to RGB
        uint8_t* d_in_frame;
        cudaMalloc((void**)&d_in_frame, H_BUFF * W_BUFF * 2 * sizeof(uint8_t));
        cudaMemcpy(d_in_frame, final_frame, H_BUFF * W_BUFF * 2 * sizeof(uint8_t), cudaMemcpyHostToDevice);
        delete[] final_frame;

        // put out frame into device memory
        uint8_t* d_out_frame;
        cudaMalloc((void**)&d_out_frame, H_BUFF * W_BUFF * 3 * sizeof(uint8_t));

        cudauyvy2bgr<<<960, 256>>>(H_BUFF * W_BUFF, d_in_frame, d_out_frame);
        cudaFree(d_in_frame);

        error = cudaGetLastError();
        if (error != cudaSuccess) {
            fprintf(stderr, "ERROR @ UYVY 2 RGB: %s \n", cudaGetErrorString(error));
            exit(EXIT_FAILURE);
        }
        
        // copy out_frame from device to host
        cudaMemcpy(frame, d_out_frame, H_BUFF * W_BUFF * 3 * sizeof(uint8_t), cudaMemcpyDeviceToHost);
        cudaFree(d_out_frame);

        cv::Mat frame_mat(H_BUFF, W_BUFF, CV_8UC3, frame);
        
        // Display the input frame
        cv::imshow("Video", frame_mat);
        if (cv::waitKey(1) == 27) {
            break;
        }

        // free memory
        delete[] frame;
        frame_mat.release();
    }

    // free memory
    free(raw_image);
    cudaFree(d_lutUYVY);
    pclose(pipe);

    // Close all windows
    cv::destroyAllWindows();
    
    return 0;
}
