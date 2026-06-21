#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace ga::utils
{

    #define GA_CUDA_CHECK(call)                                                \
        do {                                                                   \
            cudaError_t error = call;                                          \
            if (error != cudaSuccess) {                                        \
                throw std::runtime_error(                                      \
                    std::string("CUDA error: ") + cudaGetErrorString(error)    \
                );                                                             \
            }                                                                  \
        } while (0)
        
    #define CUSPARSE_CHECK(call)                                               \
        do {                                                                   \
            cusparseStatus_t status = call;                                    \
            if (status != CUSPARSE_STATUS_SUCCESS) {                           \
                fprintf(stderr, "cuSPARSE error at %s:%d: %d\n",               \
                        __FILE__, __LINE__, static_cast<int>(status));         \
                exit(EXIT_FAILURE);                                            \
            }                                                                  \
        } while (0)
        
};