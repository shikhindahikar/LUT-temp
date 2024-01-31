#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "lut.h"
#include <cstdio>
#include <cstdlib>
#include "utils.h"

int main(int argc, char* argv[]) {
    // Specify the input video file and the output directory
    std::string inputVideoFile = "/media/ozsports/ozsports/Work/ball_far.mp4";

    // Specify the size of the LUT and path to the .cube file
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <LUT size> <LUT file path>" << std::endl;
        exit(EXIT_FAILURE);
    }

    int LUT_SIZE = std::stoi(argv[1]);
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

    // interpolate the LUT values for all 256^3 RGB values
    for (int i = 0; i < 256; i++) {
        for (int j = 0; j < 256; j++){
            for (int k = 0; k < 256; k++) {
                rgb = trilinearInterpolation(make_float3(i, j, k), lutValues, LUT_SIZE, 8);
                interpolatedLUTValues[i * 256 * 256 * 3 + j * 256 * 3 + k * 3] = static_cast<uint8_t>(rgb.x);
                interpolatedLUTValues[i * 256 * 256 * 3 + j * 256 * 3 + k * 3 + 1] = static_cast<uint8_t>(rgb.y);
                interpolatedLUTValues[i * 256 * 256 * 3 + j * 256 * 3 + k * 3 + 2] = static_cast<uint8_t>(rgb.z);
            }
        }
    }

    delete[] lutValues;

    std::string ffmpeg_path = "ffmpeg";

    // Open the input video file
    cv::VideoCapture video(inputVideoFile);
    if (!video.isOpened()) {
        std::cerr << "Error opening video file: " << inputVideoFile << std::endl;
        exit(EXIT_FAILURE);
    }

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

        // put lutValues into device memory
        uint8_t* d_lutValues;
        cudaMalloc((void**)&d_lutValues, 256 * 256 * 256 * 3 * sizeof(uint8_t));
        cudaMemcpy(d_lutValues, interpolatedLUTValues, 256 * 256 * 256 * 3 * sizeof(uint8_t), cudaMemcpyHostToDevice);
        
        cudauyvy2rgb<<<960, 256>>>(H_BUFF * W_BUFF, d_frame, d_out_frame);

        // copy out_frame from device to host
        cudaMemcpy(frame, d_out_frame, H_BUFF * W_BUFF * 3 * sizeof(uint8_t), cudaMemcpyDeviceToHost);

        // apply LUTs
        cv::Mat frame_mat = applyLUTtoFrameCUDA(frame, d_lutValues, LUT_SIZE);

        // convert frame into a Mat
        // cv::Mat frame_mat(H_BUFF, W_BUFF, CV_8UC3, frame);
        // cv::cvtColor(frame_mat, frame_mat, cv::COLOR_RGB2BGR);
        // Display the input frame
        cv::imshow("Video", frame_mat);
        if (cv::waitKey(1) == 27) {
            break;
        }

        // free memory
        free(frame);
        free(out_frame);
        cudaFree(d_frame);
        cudaFree(d_out_frame);
        cudaFree(d_lutValues);
    }

    // free memory
    delete[] interpolatedLUTValues;
    free(raw_image);
    pclose(pipe);

    // Release the VideoCapture object
    video.release();
    // Close all windows
    cv::destroyAllWindows();
    
    return 0;
}
