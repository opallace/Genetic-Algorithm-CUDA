#pragma once

#include <cstddef>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "ga/core/CudaChromosome.hpp"
#include "ga/utils/CudaCheck.cuh"
#include "ga/utils/CudaRandom.cuh"

namespace ga::kernels {

    __global__
    inline void setup_rng_kernel(
        curandState* rng_states,
        std::size_t total_states,
        unsigned long seed

    ) {
        std::size_t id = blockIdx.x * blockDim.x + threadIdx.x;

        if (id >= total_states) {
            return;
        }

        curand_init(seed, id, 0, &rng_states[id]);
    }

    inline void launch_setup_rng(
        curandState* rng_states,
        std::size_t total_states,
        unsigned long seed

    ) {
        constexpr int threads = 1024;

        int blocks = static_cast<int>(
            (total_states + threads - 1) / threads
        );

        setup_rng_kernel<<<blocks, threads>>>(
            rng_states,
            total_states,
            seed
        );

        GA_CUDA_CHECK(cudaGetLastError());
        GA_CUDA_CHECK(cudaDeviceSynchronize());
    }
    
}