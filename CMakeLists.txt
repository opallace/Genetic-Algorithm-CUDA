cmake_minimum_required(VERSION 3.22)

project(GeneticAlgorithmCUDA LANGUAGES CUDA CXX)

find_package(CUDAToolkit REQUIRED)

add_executable(main
    src/main.cu

    src/ctqw/fitness/CtqwFitnessEvaluator.cu

    src/ctqw/sparse/DeviceCsrMatrix.cu
    src/ctqw/sparse/LaplacianBuilder.cpp
    src/ctqw/sparse/LaplacianStructure.cpp
    src/ctqw/sparse/LaplacianWeightUpdater.cu

    src/ctqw/solver/CtqwKernels.cu
    src/ctqw/solver/CtqwRk4Solver.cu

    src/ctqw/problem/TransferEvaluator.cu
    src/ctqw/problem/ReusableTransferEvaluator.cu
)

target_include_directories(main
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(main
    PRIVATE
        CUDA::cudart
        CUDA::curand
        CUDA::cusparse
)

target_compile_options(main
    PRIVATE
        --std=c++20
)

set_target_properties(main PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)