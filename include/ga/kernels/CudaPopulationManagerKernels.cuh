#pragma once

#include <cstddef>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "ga/core/CudaChromosome.hpp"
#include "ga/utils/CudaCheck.cuh"
#include "ga/utils/CudaRandom.cuh"

namespace ga::kernels
{
    template<
        typename GeneType,
        typename PopulationInitializer
    >
    __global__
    void initialize_population_kernel(
        GeneType* population,
        curandState* rng_states,
        std::size_t total_genes,
        PopulationInitializer initializer
    ) {
        const std::size_t gene_id = blockIdx.x * blockDim.x + threadIdx.x;

        if (gene_id >= total_genes) {
            return;
        }

        curandState rng_state = rng_states[gene_id];

        population[gene_id] = initializer.template generate<GeneType>(rng_state);

        rng_states[gene_id] = rng_state;
    }

    template<
        typename GeneType,
        typename PopulationInitializer
    >
    void launch_initialize_population(
        GeneType* population,
        curandState* rng_states,
        std::size_t total_genes,
        PopulationInitializer initializer
    ) {
        constexpr int threads = 1024;

        const int blocks = (total_genes + threads - 1) / threads;

        initialize_population_kernel<
            GeneType,
            PopulationInitializer
        ><<<blocks, threads>>>(
                population,
                rng_states,
                total_genes,
                initializer
            );

        GA_CUDA_CHECK(cudaGetLastError());
        GA_CUDA_CHECK(cudaDeviceSynchronize());
    }

    template<typename GeneType>
    __global__
    void copy_selected_individuals_kernel(
        const GeneType* source_population,
        GeneType* destination_population,
        const int* selected_indices,
        std::size_t selected_count,
        std::size_t chromosome_size
    )
    {
        const std::size_t gene_id     = blockIdx.x * blockDim.x + threadIdx.x;
        const std::size_t total_genes = selected_count * chromosome_size;

        if (gene_id >= total_genes){
            return;
        }

        const std::size_t destination_individual = gene_id / chromosome_size;
        const std::size_t locus                  = gene_id % chromosome_size;

        const int source_individual = selected_indices[destination_individual];

        destination_population[destination_individual * chromosome_size + locus] =
        source_population[static_cast<std::size_t>(source_individual) * chromosome_size + locus];
    }

    template<typename GeneType>
    void launch_copy_selected_individuals(
        const GeneType* source_population,
        GeneType* destination_population,
        const int* selected_indices,
        std::size_t selected_count,
        std::size_t chromosome_size
    )
    {
        const std::size_t total_genes = selected_count * chromosome_size;

        constexpr int threads = 1024;

        const int blocks = static_cast<int>((total_genes + threads - 1) / threads);

        copy_selected_individuals_kernel<<<blocks, threads>>>(
            source_population,
            destination_population,
            selected_indices,
            selected_count,
            chromosome_size
        );

        GA_CUDA_CHECK(cudaGetLastError());
        GA_CUDA_CHECK(cudaDeviceSynchronize());
    }

}