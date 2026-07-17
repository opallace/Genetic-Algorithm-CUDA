#pragma once

#include <cstddef>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "ga/core/CudaChromosome.hpp"
#include "ga/utils/CudaCheck.cuh"
#include "ga/utils/CudaRandom.cuh"

namespace ga::kernels {

    template<typename Selection, typename Optimization>
    __global__
    void selection_kernel(
        const double* fitness_values,
        int* parent_a_indices,
        int* parent_b_indices,
        unsigned char* pair_crosses,
        curandState* pair_rng_states,
        std::size_t population_size,
        std::size_t pair_count,
        float crossover_rate,
        Selection selection,
        Optimization optimization
    ) {
        std::size_t pair_id = blockIdx.x * blockDim.x + threadIdx.x;

        if (pair_id >= pair_count) {
            return;
        }

        curandState pair_rng_state = pair_rng_states[pair_id];

        parent_a_indices[pair_id] = selection(
            fitness_values,
            population_size,
            optimization,
            pair_rng_state
        );

        parent_b_indices[pair_id] = selection(
            fitness_values,
            population_size,
            optimization,
            pair_rng_state
        );
        
        pair_crosses[pair_id] = curand_uniform(&pair_rng_state) <= crossover_rate ? 1 : 0;

        pair_rng_states[pair_id] = pair_rng_state;
    }

    template<typename Selection, typename FitnessComparator>
    void launch_selection(
        const double* fitness_values,
        int* parent_a_indices,
        int* parent_b_indices,
        unsigned char* pair_crosses,
        curandState* pair_rng_states,
        std::size_t population_size,
        float crossover_rate,
        Selection selection,
        FitnessComparator fitness_comparator
    ) {
        constexpr int threads = 1024;

        std::size_t pair_count = population_size / 2;

        int blocks = static_cast<int>(
            (pair_count + threads - 1) / threads
        );

        selection_kernel<<<blocks, threads>>>(
            fitness_values,
            parent_a_indices,
            parent_b_indices,
            pair_crosses,
            pair_rng_states,
            population_size,
            pair_count,
            crossover_rate,
            selection,
            fitness_comparator
        );

        GA_CUDA_CHECK(cudaGetLastError());
        GA_CUDA_CHECK(cudaDeviceSynchronize());
    }
    
}