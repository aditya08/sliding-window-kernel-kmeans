cmake_minimum_required(VERSION 3.20)

# Set the project name and enable CUDA language
project(StreamingKernelKMeans LANGUAGES CUDA CXX)

# Set the C++ standard to 17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_ARCHITECTURES 80)

# Find CUDA Toolkit (required)
find_package(CUDAToolkit REQUIRED)

# Add subdirectories
add_subdirectory(src)
add_subdirectory(test)
