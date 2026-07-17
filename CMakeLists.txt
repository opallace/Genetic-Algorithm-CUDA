cmake_minimum_required(VERSION 3.22)

project(GeneticAlgorithmCUDA LANGUAGES CUDA CXX)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Para GTX 1060, se quiser fixar:
set(CMAKE_CUDA_ARCHITECTURES 61)

# ============================================================
# Biblioteca genérica do GA
# ============================================================

add_library(ga_cuda INTERFACE)

target_include_directories(ga_cuda
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(ga_cuda
    INTERFACE
        CUDA::cudart
        CUDA::curand
)

# ============================================================
# Função para problemas simples que só usam o GA
# ============================================================

function(add_ga_problem target_name main_file)
    add_executable(${target_name}
        ${main_file}
    )

    target_include_directories(${target_name}
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    target_link_libraries(${target_name}
        PRIVATE
            ga_cuda
    )

    target_compile_options(${target_name}
        PRIVATE
            --std=c++20
    )

    set_target_properties(${target_name} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
endfunction()

# ============================================================
# Problema Target 5
# ============================================================

add_ga_problem(target_5
    problems/target_5/main.cu
)

# ============================================================
# Problema Sphere
# ============================================================

add_ga_problem(sphere
    problems/sphere/main.cu
)

# ============================================================
# Problema CTQW
# ============================================================

add_executable(ctqw
    problems/ctqw/main.cu

    problems/ctqw/src/fitness/CtqwFitnessEvaluator.cu

    problems/ctqw/src/solver/CtqwBatchKernels.cu

    problems/ctqw/src/sparse/DeviceWeightedLaplacian.cu
)

target_include_directories(ctqw
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/problems/ctqw/include
)

target_link_libraries(ctqw
    PRIVATE
        ga_cuda
)

target_compile_options(ctqw
    PRIVATE
        --std=c++20
)

set_target_properties(ctqw PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# ============================================================
# Problema Suspended Load
# ============================================================

add_executable(suspended_load
    problems/suspended_load/main.cu

    problems/suspended_load/src/solver/SuspendedLoadRk4Solver.cu

)

target_include_directories(suspended_load
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/problems/suspended_load/include
)

target_link_libraries(suspended_load
    PRIVATE
        ga_cuda
)

target_compile_options(suspended_load
    PRIVATE
        --std=c++20
)

set_target_properties(suspended_load PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)