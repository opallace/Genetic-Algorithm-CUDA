cmake_minimum_required(VERSION 3.22)

project(GeneticAlgorithmCUDA LANGUAGES CUDA)

find_package(CUDAToolkit REQUIRED)

add_executable(main
    src/main.cu
)

target_include_directories(main
    PRIVATE
        include
)

target_link_libraries(main
    PRIVATE
        CUDA::cudart
        CUDA::curand
)

target_compile_options(main
    PRIVATE
        --std=c++20
)

set_target_properties(main PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)