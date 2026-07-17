#pragma once

#include <cstddef>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "ga/core/CudaChromosome.hpp"
#include "ga/utils/CudaCheck.cuh"
#include "ga/utils/CudaRandom.cuh"

namespace ga::kernels {

    template<
        typename GeneType,
        typename Crossover,
        typename Mutation
    >
    __global__
    void crossover_mutation_kernel(
        const GeneType* population,
        GeneType* next_population,
        const int* parent_a_indices,
        const int* parent_b_indices,
        const unsigned char* pair_crosses,
        curandState* gene_rng_states,
        std::size_t chromosome_size,
        std::size_t offspring_count,
        Crossover crossover,
        Mutation mutation
    ) {
        std::size_t pair_count       = offspring_count / 2;
        std::size_t pair_gene_id     = blockIdx.x * blockDim.x + threadIdx.x;
        std::size_t total_pair_genes = pair_count * chromosome_size;

        if (pair_gene_id >= total_pair_genes) {
            return;
        }

        std::size_t pair_id = pair_gene_id / chromosome_size;
        std::size_t locus   = pair_gene_id % chromosome_size;

        std::size_t child_a_id = 2 * pair_id;
        std::size_t child_b_id = child_a_id + 1;

        int parent_a = parent_a_indices[pair_id];
        int parent_b = parent_b_indices[pair_id];

        GeneType allele_a = population[parent_a * chromosome_size + locus];
        GeneType allele_b = population[parent_b * chromosome_size + locus];

        curandState gene_rng_state = gene_rng_states[pair_gene_id];

        GeneType child_a_allele;
        GeneType child_b_allele;

        if (pair_crosses[pair_id]) {
            crossover(allele_a, allele_b, child_a_allele, child_b_allele, gene_rng_state);

        } else {
            child_a_allele = allele_a;
            child_b_allele = allele_b;
        }

        child_a_allele = mutation(child_a_allele, gene_rng_state);
        child_b_allele = mutation(child_b_allele, gene_rng_state);

        next_population[child_a_id * chromosome_size + locus] = child_a_allele;
        next_population[child_b_id * chromosome_size + locus] = child_b_allele;

        gene_rng_states[pair_gene_id] = gene_rng_state;
    }

    template<
        typename GeneType,
        typename Crossover,
        typename Mutation
    >
    void launch_crossover_mutation(
        const GeneType* population,
        GeneType* next_population,
        const int* parent_a_indices,
        const int* parent_b_indices,
        const unsigned char* pair_crosses,
        curandState* gene_rng_states,
        std::size_t chromosome_size,
        std::size_t offspring_count,
        Crossover crossover,
        Mutation mutation
    ) {
        std::size_t pair_count       = offspring_count / 2;
        std::size_t total_pair_genes = pair_count * chromosome_size;

        constexpr int threads = 1024;

        int blocks = static_cast<int>(
            (total_pair_genes + threads - 1) / threads
        );

        crossover_mutation_kernel<<<blocks, threads>>>(
            population,
            next_population,
            parent_a_indices,
            parent_b_indices,
            pair_crosses,
            gene_rng_states,
            chromosome_size,
            offspring_count,
            crossover,
            mutation
        );

        GA_CUDA_CHECK(cudaGetLastError());
        GA_CUDA_CHECK(cudaDeviceSynchronize());
    }
    
}