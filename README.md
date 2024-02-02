# LUT Transformation using trilinear interpolation algorithm

This project is aimed at fetching transformation data from a LUT .cube file and interpolating for all possible 0 to 255 RGB values. 
Later on we can apply the LUT transorms very quickly to any frame. For this example I am using a video to apply LUT frame by frame.

## Dependencies

- FFMPEG
  - `sudo apt update`
  - `sudo apt install ffmpeg`
- OpenCV
  - Install OpenCV from [here](https://opencv.org/get-started/)
- CUDA[Optional since you can modify the code to not use CUDA if you wish]
  - CUDA supporting graphics card
  - CUDA Toolkit
