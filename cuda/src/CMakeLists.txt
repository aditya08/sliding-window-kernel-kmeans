# Find all source files in the current directory
file(GLOB SOURCES "*.cpp" "*.cu")

# Create a library target
add_library(kernel_kmeans STATIC ${SOURCES})

target_link_libraries(kernel_kmeans CUDA::cublas)
target_include_directories(kernel_kmeans PUBLIC ${PROJECT_SOURCE_DIR}/include)