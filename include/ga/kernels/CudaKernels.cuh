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

    __global__
    void create_uniform_population_kernel(
        float* population,
        curandState* rng_states,
        std::size_t total_genes,
        float min_value,
        float max_value
    ) {
        std::size_t gene_id = blockIdx.x * blockDim.x + threadIdx.x;

        if (gene_id >= total_genes) {
            return;
        }

        curandState rng_state = rng_states[gene_id];

        float u = curand_uniform(&rng_state);

        population[gene_id] = min_value + (max_value - min_value) * u;

        rng_states[gene_id] = rng_state;
    }

    __global__
    void create_truncated_gaussian_population_kernel(
        float* population,
        curandState* rng_states,
        std::size_t total_genes,
        float min_value,
        float max_value,
        float sigma
    ) {
        std::size_t gene_id = blockIdx.x * blockDim.x + threadIdx.x;

        if (gene_id >= total_genes) {
            return;
        }

         curandState rng_state = rng_states[gene_id];

        float mean = 0.5f * (min_value + max_value);
        float alpha = (min_value - mean) / sigma;
        float beta  = (max_value - mean) / sigma;

        float phi_alpha = ga::utils::standard_normal_cdf(alpha);
        float phi_beta  = ga::utils::standard_normal_cdf(beta);

        float u = curand_uniform(&rng_state);
        float p = phi_alpha + (phi_beta - phi_alpha) * u;
        float z = ga::utils::standard_normal_inverse_cdf(p);

        float value = mean + sigma * z;

        population[gene_id] = value;

        rng_states[gene_id] = rng_state;
    }

    void launch_create_uniform_population(
        float* population,
        curandState* rng_states,
        std::size_t total_genes,
        float min_value,
        float max_value
    ) {
        constexpr int threads = 1024;

        int blocks = static_cast<int>(
            (total_genes + threads - 1) / threads
        );

        create_uniform_population_kernel<<<blocks, threads>>>(
            population,
            rng_states,
            total_genes,
            min_value,
            max_value
        );

        GA_CUDA_CHECK(cudaGetLastError());
        GA_CUDA_CHECK(cudaDeviceSynchronize());
    }

    void launch_create_truncated_gaussian_population(
        float* population,
        curandState* rng_states,
        std::size_t total_genes,
        float min_value,
        float max_value,
        float sigma
    ) {
        constexpr int threads = 1024;

        int blocks = static_cast<int>(
            (total_genes + threads - 1) / threads
        );

        create_truncated_gaussian_population_kernel<<<blocks, threads>>>(
            population,
            rng_states,
            total_genes,
            min_value,
            max_value,
            sigma
        );

        GA_CUDA_CHECK(cudaGetLastError());
        GA_CUDA_CHECK(cudaDeviceSynchronize());
    }

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

    template<typename Optimization>
    __device__
    int select_parent_tournament_device(
        const double* fitness_values,
        std::size_t population_size,
        std::size_t tournament_size,
        Optimization optimization,
        curandState& rng_state
    ) {
        int best_index = static_cast<int>(
            curand(&rng_state) % population_size
        );

        double best_fitness = fitness_values[best_index];

        for (std::size_t i = 1; i < tournament_size; i++) {
            int candidate_index = static_cast<int>(
                curand(&rng_state) % population_size
            );

            double candidate_fitness = fitness_values[candidate_index];

            if (optimization.is_better(candidate_fitness, best_fitness)) {
                best_index = candidate_index;
                best_fitness = candidate_fitness;
            }
        }

        return best_index;
    }

    template<typename Selection, typename Optimization>
    __global__
    void selection_kernel(
        const double* fitness_values,
        int* parent_a_indices,
        int* parent_b_indices,
        curandState* rng_states,
        std::size_t population_size,
        std::size_t pair_count,
        Selection selection,
        Optimization optimization
    ) {
        std::size_t pair_id = blockIdx.x * blockDim.x + threadIdx.x;

        if (pair_id >= pair_count) {
            return;
        }

        curandState rng_state = rng_states[pair_id];

        parent_a_indices[pair_id] =
            select_parent_tournament_device(
                fitness_values,
                population_size,
                selection.tournament_size,
                optimization,
                rng_state
            );

        parent_b_indices[pair_id] =
            select_parent_tournament_device(
                fitness_values,
                population_size,
                selection.tournament_size,
                optimization,
                rng_state
            );

        rng_states[pair_id] = rng_state;
    }

    template<typename Selection, typename Optimization>
    void launch_selection(
        const double* fitness_values,
        int* parent_a_indices,
        int* parent_b_indices,
        curandState* rng_states,
        std::size_t population_size,
        Selection selection,
        Optimization optimization
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
            rng_states,
            population_size,
            pair_count,
            selection,
            optimization
        );

        GA_CUDA_CHECK(cudaGetLastError());
        GA_CUDA_CHECK(cudaDeviceSynchronize());
    }

    template<
        typename GeneType,
        typename Crossover,
        typename Mutation
    >
    __global__
    void crossover_mutation_kernel(
        std::size_t generation,
        const GeneType* population,
        GeneType* next_population,
        const int* parent_a_indices,
        const int* parent_b_indices,
        curandState* rng_states,
        std::size_t population_size,
        std::size_t chromosome_size,
        Crossover crossover,
        Mutation mutation
    ) {
        std::size_t gene_id = blockIdx.x * blockDim.x + threadIdx.x;
        std::size_t total_genes = population_size * chromosome_size;

        if (gene_id >= total_genes) {
            return;
        }

        std::size_t child_id = gene_id / chromosome_size;
        std::size_t locus = gene_id % chromosome_size;
        std::size_t pair_id = child_id / 2;

        int parent_a = parent_a_indices[pair_id];
        int parent_b = parent_b_indices[pair_id];

        GeneType allele_a = population[parent_a * chromosome_size + locus];
        GeneType allele_b = population[parent_b * chromosome_size + locus];

        curandState rng_state = rng_states[gene_id];

        GeneType child_allele = crossover(allele_a, allele_b, rng_state);
                 child_allele = mutation(generation, child_allele, rng_state);

        next_population[gene_id] = child_allele;

        rng_states[gene_id] = rng_state;
    }

    template<
        typename GeneType,
        typename Crossover,
        typename Mutation
    >
    void launch_crossover_mutation(
        std::size_t generation,
        const GeneType* population,
        GeneType* next_population,
        const int* parent_a_indices,
        const int* parent_b_indices,
        curandState* rng_states,
        std::size_t population_size,
        std::size_t chromosome_size,
        Crossover crossover,
        Mutation mutation
    ) {
        std::size_t total_genes = population_size * chromosome_size;

        constexpr int threads = 1024;

        int blocks = static_cast<int>(
            (total_genes + threads - 1) / threads
        );

        crossover_mutation_kernel<<<blocks, threads>>>(
            generation,
            population,
            next_population,
            parent_a_indices,
            parent_b_indices,
            rng_states,
            population_size,
            chromosome_size,
            crossover,
            mutation
        );

        GA_CUDA_CHECK(cudaGetLastError());
        GA_CUDA_CHECK(cudaDeviceSynchronize());
    }

}