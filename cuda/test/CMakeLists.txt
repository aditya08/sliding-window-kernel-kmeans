# Add test executables
add_executable(test_kernel_function test_kernel_function.cu)

# Link CUDA libraries to the test executables
target_link_libraries(test_kernel_function PRIVATE kernel_kmeans)

# Add tests
add_test(NAME TestKernelFunction COMMAND test_kernel_function)