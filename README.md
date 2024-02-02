# LUT Transformation using trilinear interpolation algorithm

This project is aimed at fetching transformation data from a LUT .cube file and interpolating for all possible 0 to 255 RGB values. 
Later on we can apply the LUT transorms very quickly to any frame. For this example I am using a video to apply LUT frame by frame.

We can skip the part where I am getting the video in uyvy colour space and instead put `-pix_fmt rgb24` for direct RGB colour spaces.
Doing this you will not have to convert uyvy to RGB later on in the code.

Used [this](https://github.com/Ppkturner/3DLutInterpolation/blob/master/LutInterpolation/Trilinear.cpp) repo code for trilinear interpolation algorithm

## Dependencies

- FFMPEG
  - `sudo apt update`
  - `sudo apt install ffmpeg`
- OpenCV
  - Install OpenCV from [here](https://opencv.org/get-started/)
- CUDA[Optional. You can modify the code to not use CUDA if you wish.]
  - CUDA supporting graphics card
  - CUDA Toolkit


## Building and Running

- Create a build directory inside the source and run following commands in that build directory:
  
  `cmake ..`

  `make`
- To run use the following commands:
  `./lut <size of the LUT> /path/to/the/cube/file`

