cmake_minimum_required(VERSION 3.13)  # CMake version check
project(lut LANGUAGES CXX CUDA)               # Create project "test"
set(CMAKE_CXX_STANDARD 14)            # Enable c++14 standard
# add opencv path
# set(OpenCV_DIR /usr/local/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)
enable_language(CUDA)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")
# Add main.cpp file of the project root directory as a source file
# Add executable target with source files listed in SOURCE_FILES variable
set(SOURCE_FILES main.cu lut.cu utils.cu)
add_executable(lut ${SOURCE_FILES})
set_property(TARGET lut PROPERTY CXX_STANDARD 14 CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(lut PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(lut ${OpenCV_LIBS}) 