#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>

#include "kernel_function.cuh"


template <typename T>
int test_kernel(const KernelType &kernel_type, const T tol = 1e-4) {
    std::unique_ptr<KernelFunction<T>> kern;
    if (kernel_type == KernelType::LINEAR) {
        kern = std::make_unique<LinearKernel<T>>();
    } else if (kernel_type == KernelType::RBF) {
        kern = std::make_unique<RBFKernel<T>>(0.5f);
    } else if (kernel_type == KernelType::TANH) {
        kern = std::make_unique<TanhKernel<T>>(1.0f, 0.0f);
    } else if (kernel_type == KernelType::POLYNOMIAL) {
        kern = std::make_unique<PolynomialKernel<T>>(1.0f, 0.0f, 3);
    } else {
        std::cerr << "Unknown kernel type: " << kernel_type << std::endl;
        return EXIT_FAILURE;
    }
    int m = 5, n = 3;
    const T A[] = {0.823457828327293,0.694828622975817,0.317099480060861,0.950222048838355,0.034446080502909,0.438744359656398,0.381558457093008,0.765516788149002,0.795199901137063,0.186872604554379,0.489764395788231,0.445586200710899,0.646313010111265,0.709364830858073,0.754686681982361};
    T C[25] = {0};

    std::unordered_map<KernelType, T*> ref_solutions = {
        {KernelType::LINEAR, new T[25]{1.110448571545615,0.957800946226812,0.913525323124503,1.478778893966979,0.479972862678362,0.957800946226812,0.826920733749641,0.800407358287468,1.279739904973963,0.431514916701106,0.913525323124503,0.800407358287468,1.104288540231916,1.368525510982806,0.641740771372016,1.478778893966979,1.279739904973963,1.368525510982806,2.038463288125258,0.716680692372050,0.479972862678362,0.431514916701106,0.641740771372016,0.716680692372050,0.605659890756495}},
        {KernelType::RBF, new T[25]{1.000000000000000,0.989175306823598,0.823787036963046,0.908757464744841,0.685174746774924,0.989175306823598,1.000000000000000,0.847726449215882,0.858170822091844,0.752183179035802,0.823787036963046,0.847726449215882,1.000000000000000,0.816400363176165,0.807967499935433,0.908757464744841,0.858170822091844,0.816400363176165,1.000000000000000,0.545866468099155,0.685174746774924,0.752183179035802,0.807967499935433,0.545866468099155,1.000000000000000}},
        {KernelType::TANH, new T[25]{0.804220896941090,0.743294368673147,0.722819999705960,0.901238954938154,0.446221876609987,0.743294368673147,0.678819120146524,0.664264444424923,0.856415601981854,0.406586568876872,0.722819999705960,0.664264444424923,0.802034191449996,0.878355721582053,0.566083658446903,0.901238954938154,0.856415601981854,0.878355721582053,0.966646637869479,0.614849036414936,0.446221876609987,0.406586568876872,0.566083658446903,0.614849036414936,0.541064771739789}},
        {KernelType::POLYNOMIAL, new T[25]{1.369289725145034,0.878669972705615,0.762362932226127,3.233774486698771,0.110573243743729,0.878669972705615,0.565446661125944,0.512782526237394,2.095873840684209,0.080350288285464,0.762362932226127,0.512782526237394,1.346628172120034,2.563059527109584,0.264288883284108,3.233774486698771,2.095873840684209,2.563059527109584,8.470492908235624,0.368109574660292,0.110573243743729,0.080350288285464,0.264288883284108,0.368109574660292,0.222170525182088}}
    };
    std::cout << "Computing Kernel Matrix for " << kern->get_kernel_name() << "\n";
    kern->compute_kernel_matrix(m, m, n, A, m, A, m, C, m);
    T* ref_C = ref_solutions[kernel_type];
    T error = 0.0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            // std::cout << C[i * m + j] << " ";
            error = std::abs(C[i * m + j] - ref_C[i * m + j]);
            if (error > tol) {
                std::cerr << "Test failed for " << kern->get_kernel_name() << " at (" << i << ", " << j << "), with absolute error: " << error << std::endl;
                return EXIT_FAILURE;
            }
        }
        // std::cout << std::endl;
    }
    return EXIT_SUCCESS;
}

int main(int argc, char** argv) {
    // single-precision tests
    for (auto kernel_type : {KernelType::LINEAR, KernelType::RBF, KernelType::TANH, KernelType::POLYNOMIAL}) {
        int single_precision_test = test_kernel<float>(kernel_type);
        int double_precision_test = test_kernel<double>(kernel_type, 1e-6);
        if (single_precision_test != EXIT_SUCCESS) {
            std::cerr << "Single-precision test failed for kernel type: " << kernel_type << std::endl;
            return EXIT_FAILURE;
        }
        if (double_precision_test != EXIT_SUCCESS) {
            std::cerr << "Double-precision test failed for kernel type: " << kernel_type << std::endl;
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}

template int test_kernel<float>(const KernelType &kernel_type, const float tol);
template int test_kernel<double>(const KernelType &kernel_type, const double tol);
