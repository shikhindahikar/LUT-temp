#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "lut.h"
#include <cstdlib>
#include "utils.h"
#include <memory>

int main(int argc, char* argv[]) {
    // Specify the input video file and the output directory
    std::string inputVideoFile = "/path/to/your/video.mp4";

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

    // put lutValues into device memory
    uint8_t* d_lutValues;
    cudaMalloc((void**)&d_lutValues, 256 * 256 * 256 * 3 * sizeof(uint8_t));
    cudaMemcpy(d_lutValues, interpolatedLUTValues, 256 * 256 * 256 * 3 * sizeof(uint8_t), cudaMemcpyHostToDevice);

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
        uint8_t* frame = (uint8_t*)malloc(H_BUFF * W_BUFF * 3 * sizeof(uint8_t));
        uint8_t* raw_video_data = (uint8_t*)raw_image;
        uint8_t* out_frame = (uint8_t*)malloc(H_BUFF * W_BUFF * 3 * sizeof(uint8_t));
        // put frame into device memory
        uint8_t* d_frame;
        cudaMalloc((void**)&d_frame, H_BUFF * W_BUFF * 2 * sizeof(uint8_t));
        cudaMemcpy(d_frame, raw_video_data, H_BUFF * W_BUFF * 2 * sizeof(uint8_t), cudaMemcpyHostToDevice);

        // put out frame into device memory
        uint8_t* d_out_frame;
        cudaMalloc((void**)&d_out_frame, H_BUFF * W_BUFF * 3 * sizeof(uint8_t));
        cudaMemcpy(d_out_frame, out_frame, H_BUFF * W_BUFF * 3 * sizeof(uint8_t), cudaMemcpyHostToDevice);
        free(out_frame);
        
        cudauyvy2rgb<<<960, 256>>>(H_BUFF * W_BUFF, d_frame, d_out_frame);
        cudaFree(d_frame);
        
        // copy out_frame from device to host
        cudaMemcpy(frame, d_out_frame, H_BUFF * W_BUFF * 3 * sizeof(uint8_t), cudaMemcpyDeviceToHost);
        cudaFree(d_out_frame);

        // apply LUTs
        uint8_t* final_frame = applyLUTtoFrameCUDA(frame, d_lutValues, LUT_SIZE);
        cv::Mat frame_mat(H_BUFF, W_BUFF, CV_8UC3, final_frame);
        delete[] final_frame;
        
        // Display the input frame
        cv::imshow("Video", frame_mat);
        if (cv::waitKey(1) == 27) {
            break;
        }

        // free memory
        free(frame);
        frame_mat.release();
    }

    // free memory
    free(raw_image);
    delete[] interpolatedLUTValues;
    cudaFree(d_lutValues);
    pclose(pipe);

    // Close all windows
    cv::destroyAllWindows();
    
    return 0;
}
