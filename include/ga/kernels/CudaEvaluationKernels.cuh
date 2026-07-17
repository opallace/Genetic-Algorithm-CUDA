#pragma once

#include <cstddef>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "ga/core/CudaChromosome.hpp"
#include "ga/utils/CudaCheck.cuh"
#include "ga/utils/CudaRandom.cuh"

namespace ga::kernels {

    template<typename GeneType, typename Fitness>
    __global__
    void evaluate_fitness_kernel(
        const GeneType* population,
        double* fitness_values,
        std::size_t population_size,
        std::size_t chromosome_size,
        Fitness fitness
    ) {
        std::size_t individual_id = blockIdx.x * blockDim.x + threadIdx.x;

        if (individual_id >= population_size) {
            return;
        }

        auto chromosome = ga::core::CudaChromosome<const GeneType>{
            &population[individual_id * chromosome_size],
            chromosome_size
        };

        fitness_values[individual_id] = fitness(chromosome);
    }

    template<typename GeneType, typename Fitness>
    void launch_evaluate_fitness(
        const GeneType* population,
        double* fitness_values,
        std::size_t population_size,
        std::size_t chromosome_size,
        Fitness fitness
    ) {
        constexpr int threads = 1024;

        int blocks = static_cast<int>(
            (population_size + threads - 1) / threads
        );

        evaluate_fitness_kernel<<<blocks, threads>>>(
            population,
            fitness_values,
            population_size,
            chromosome_size,
            fitness
        );

        GA_CUDA_CHECK(cudaGetLastError());
        GA_CUDA_CHECK(cudaDeviceSynchronize());
    }
    
}