#pragma once

#include <cstddef>
#include <vector>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <utility>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "ga/cuda/utils/CudaCheck.hpp"
#include "ga/cuda/kernels/CudaKernels.cuh"
#include "ga/cuda/concepts/CudaMutationConcept.hpp"
#include "ga/cuda/concepts/CudaCrossoverConcept.hpp"
#include "ga/cuda/concepts/CudaFitnessConcept.hpp"
#include "ga/cuda/concepts/CudaSelectionConcept.hpp"

namespace ga::cuda::core {

template<
    typename GeneType,
    typename Mutation,
    typename Crossover,
    typename Selection,
    typename Fitness,
    typename Optimization
>
requires
    ga::cuda::concepts::CudaMutationConcept<Mutation, GeneType>   &&
    ga::cuda::concepts::CudaCrossoverConcept<Crossover, GeneType> &&
    ga::cuda::concepts::CudaSelectionConcept<Selection>           &&
    ga::cuda::concepts::CudaFitnessConcept<
        Fitness,
        ga::cuda::core::CudaChromosome<const GeneType>
    >
class CudaGeneticAlgorithm {
public:
    CudaGeneticAlgorithm(
        std::size_t population_size,
        std::size_t chromosome_size,
        Mutation mutation,
        Crossover crossover,
        Selection selection,
        Fitness fitness,
        Optimization optimization
    )
        : population_size(population_size),
          chromosome_size(chromosome_size),
          total_genes(population_size * chromosome_size),
          pair_count(population_size / 2),
          mutation(std::move(mutation)),
          crossover(std::move(crossover)),
          selection(std::move(selection)),
          fitness(std::move(fitness)),
          optimization(std::move(optimization))
    {
        if (population_size < 2) {
            throw std::invalid_argument(
                "Population size must be at least 2."
            );
        }

        if (population_size % 2 != 0) {
            throw std::invalid_argument(
                "CUDA GA currently requires even population size."
            );
        }

        if (chromosome_size == 0) {
            throw std::invalid_argument(
                "Chromosome size must be greater than 0."
            );
        }

        allocate();
    }

    ~CudaGeneticAlgorithm() {
        release();
    }

    CudaGeneticAlgorithm(const CudaGeneticAlgorithm&) = delete;
    CudaGeneticAlgorithm& operator=(const CudaGeneticAlgorithm&) = delete;

    void initialize_random(
        GeneType min_value,
        GeneType max_value,
        unsigned long seed = 1234
    ) {
        ga::cuda::kernels::launch_setup_rng(
            d_rng_states,
            total_genes,
            seed
        );

        ga::cuda::kernels::launch_initialize_population(
            d_population,
            d_rng_states,
            total_genes,
            min_value,
            max_value
        );
    }

    void evaluate() {
        ga::cuda::kernels::launch_evaluate_fitness(
            d_population,
            d_fitness_values,
            population_size,
            chromosome_size,
            fitness
        );
    }

    void select() {
        ga::cuda::kernels::launch_selection(
            d_fitness_values,
            d_parent_a_indices,
            d_parent_b_indices,
            d_rng_states,
            population_size,
            pair_count,
            selection,
            optimization
        );
    }

    void reproduce() {
        ga::cuda::kernels::launch_crossover_mutation(
            d_population,
            d_next_population,
            d_parent_a_indices,
            d_parent_b_indices,
            d_rng_states,
            population_size,
            chromosome_size,
            crossover,
            mutation
        );

        std::swap(d_population, d_next_population);
    }

    void run(std::size_t generations) {
        evaluate();
        //print_best(0);

        for (std::size_t generation = 1; generation <= generations; generation++) {
            select();
            reproduce();
            evaluate();

            //print_best(generation);
        }
    }

    std::vector<double> copy_fitness_to_host() const {
        std::vector<double> host_fitness(population_size);

        GA_CUDA_CHECK(cudaMemcpy(
            host_fitness.data(),
            d_fitness_values,
            population_size * sizeof(double),
            cudaMemcpyDeviceToHost
        ));

        return host_fitness;
    }

    std::vector<GeneType> copy_population_to_host() const {
        std::vector<GeneType> host_population(total_genes);

        GA_CUDA_CHECK(cudaMemcpy(
            host_population.data(),
            d_population,
            total_genes * sizeof(GeneType),
            cudaMemcpyDeviceToHost
        ));

        return host_population;
    }

    std::size_t best_index_host() const {
        auto fitness_values = copy_fitness_to_host();

        std::size_t best_index = 0;

        for (std::size_t i = 1; i < population_size; i++) {
            if (optimization.is_better(
                    fitness_values[i],
                    fitness_values[best_index]
                )
            ) {
                best_index = i;
            }
        }

        return best_index;
    }

    double best_fitness_host() const {
        auto fitness_values = copy_fitness_to_host();
        return fitness_values[best_index_host()];
    }

    void print_best(std::size_t generation) const {
        auto fitness_values = copy_fitness_to_host();
        auto population = copy_population_to_host();

        std::size_t best_index = 0;

        for (std::size_t i = 1; i < population_size; i++) {
            if (optimization.is_better(
                    fitness_values[i],
                    fitness_values[best_index]
                )
            ) {
                best_index = i;
            }
        }

        std::cout << "Generation " << generation
                  << " | best fitness = "
                  << fitness_values[best_index]
                  << " | individual = [";

        std::size_t offset = best_index * chromosome_size;

        for (std::size_t locus = 0; locus < chromosome_size; locus++) {
            std::cout << std::fixed
                      << std::setprecision(6)
                      << std::setw(10)
                      << population[offset + locus];

            if (locus + 1 < chromosome_size) {
                std::cout << ",";
            }
        }

        std::cout << "]\n";
    }

private:
    void allocate() {
        GA_CUDA_CHECK(cudaMalloc(
            &d_population,
            total_genes * sizeof(GeneType)
        ));

        GA_CUDA_CHECK(cudaMalloc(
            &d_next_population,
            total_genes * sizeof(GeneType)
        ));

        GA_CUDA_CHECK(cudaMalloc(
            &d_fitness_values,
            population_size * sizeof(double)
        ));

        GA_CUDA_CHECK(cudaMalloc(
            &d_parent_a_indices,
            pair_count * sizeof(int)
        ));

        GA_CUDA_CHECK(cudaMalloc(
            &d_parent_b_indices,
            pair_count * sizeof(int)
        ));

        GA_CUDA_CHECK(cudaMalloc(
            &d_rng_states,
            total_genes * sizeof(curandState)
        ));
    }

    void release() {
        if (d_population) {
            cudaFree(d_population);
        }

        if (d_next_population) {
            cudaFree(d_next_population);
        }

        if (d_fitness_values) {
            cudaFree(d_fitness_values);
        }

        if (d_parent_a_indices) {
            cudaFree(d_parent_a_indices);
        }

        if (d_parent_b_indices) {
            cudaFree(d_parent_b_indices);
        }

        if (d_rng_states) {
            cudaFree(d_rng_states);
        }
    }

private:
    std::size_t population_size;
    std::size_t chromosome_size;
    std::size_t total_genes;
    std::size_t pair_count;

    Mutation mutation;
    Crossover crossover;
    Selection selection;
    Fitness fitness;
    Optimization optimization;

    GeneType* d_population = nullptr;
    GeneType* d_next_population = nullptr;

    double* d_fitness_values = nullptr;

    int* d_parent_a_indices = nullptr;
    int* d_parent_b_indices = nullptr;

    curandState* d_rng_states = nullptr;
};

}