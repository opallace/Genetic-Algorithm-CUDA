#pragma once

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>
#include <cusparse.h>

#define CTQW_CUDA_CHECK(call)                                             \
    do {                                                                  \
        cudaError_t err = (call);                                         \
        if (err != cudaSuccess) {                                         \
            std::fprintf(                                                 \
                stderr,                                                   \
                "CUDA error at %s:%d: %s\n",                              \
                __FILE__,                                                 \
                __LINE__,                                                 \
                cudaGetErrorString(err)                                   \
            );                                                            \
            std::exit(EXIT_FAILURE);                                      \
        }                                                                 \
    } while (0)


#define CTQW_CUSPARSE_CHECK(call)                                         \
    do {                                                                  \
        cusparseStatus_t status = (call);                                 \
        if (status != CUSPARSE_STATUS_SUCCESS) {                          \
            std::fprintf(                                                 \
                stderr,                                                   \
                "cuSPARSE error at %s:%d: %d\n",                          \
                __FILE__,                                                 \
                __LINE__,                                                 \
                static_cast<int>(status)                                  \
            );                                                            \
            std::exit(EXIT_FAILURE);                                      \
        }                                                                 \
    } while (0)
