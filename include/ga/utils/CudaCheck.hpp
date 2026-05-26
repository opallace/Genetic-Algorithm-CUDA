#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#define GA_CUDA_CHECK(call)                                                \
    do {                                                                   \
        cudaError_t error = call;                                          \
        if (error != cudaSuccess) {                                        \
            throw std::runtime_error(                                      \
                std::string("CUDA error: ") + cudaGetErrorString(error)    \
            );                                                             \
        }                                                                  \
    } while (0)