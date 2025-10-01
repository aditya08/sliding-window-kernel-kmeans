# Add test executables
add_executable(test_kernel_kmeans test_kernel_kmeans.cu)
add_executable(test_kernel_function test_kernel_function.cu)
add_executable(test_dataloader test_dataloader.cpp)

# Link CUDA libraries to the test executables
target_link_libraries(test_kernel_kmeans PRIVATE kernel_kmeans)
target_link_libraries(test_kernel_function PRIVATE kernel_kmeans)
target_link_libraries(test_dataloader PRIVATE kernel_kmeans)

# copy files
file(COPY ${CMAKE_CURRENT_LIST_DIR}/data DESTINATION ${CMAKE_BINARY_DIR}/test)

# Add tests
add_test(NAME TestKernelKmeans COMMAND test_kernel_kmeans)
add_test(NAME TestKernelFunction COMMAND test_kernel_function)
add_test(NAME TestDataLoader COMMAND test_dataloader)