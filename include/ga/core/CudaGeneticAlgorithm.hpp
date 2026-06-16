#pragma once

#include <cstddef>
#include <vector>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <utility>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "ga/utils/CudaCheck.cuh"
#include "ga/kernels/CudaKernels.cuh"
#include "ga/concepts/CudaMutationConcept.hpp"
#include "ga/concepts/CudaCrossoverConcept.hpp"
#include "ga/concepts/CudaSelectionConcept.hpp"

namespace ga::core {

    template<
        typename GeneType,
        typename Mutation,
        typename Crossover,
        typename Selection,
        typename Optimization
    >
    requires
        ga::concepts::CudaMutationConcept<Mutation, GeneType>   &&
        ga::concepts::CudaCrossoverConcept<Crossover, GeneType> &&
        ga::concepts::CudaSelectionConcept<Selection>           
    class CudaGeneticAlgorithm {
    public:
        CudaGeneticAlgorithm(
            std::size_t population_size,
            std::size_t chromosome_size,
            Mutation mutation,
            Crossover crossover,
            Selection selection,
            Optimization optimization
        )
            : population_size(population_size),
            chromosome_size(chromosome_size),
            total_genes(population_size * chromosome_size),
            pair_count(population_size / 2),
            mutation(std::move(mutation)),
            crossover(std::move(crossover)),
            selection(std::move(selection)),
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
            ga::kernels::launch_setup_rng(
                d_rng_states,
                total_genes,
                seed
            );

            ga::kernels::launch_initialize_population(
                d_population,
                d_rng_states,
                total_genes,
                min_value,
                max_value
            );
        }

        template<typename Fitness>
        void evaluate(Fitness fitness) {
            ga::kernels::launch_evaluate_fitness(
                d_population,
                d_fitness_values,
                population_size,
                chromosome_size,
                fitness
            );
        }

        template<typename PopulationEvaluator>
        void evaluate_with(PopulationEvaluator& evaluator) {
            evaluator.evaluate_population(
                d_population,
                d_fitness_values,
                population_size,
                chromosome_size
            );
        }

        void select() {
            ga::kernels::launch_selection(
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

        void reproduce(std::size_t generation) {
            ga::kernels::launch_crossover_mutation(
                generation,
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

        template<typename Fitness>
        void run(std::size_t generations, Fitness fitness) {
            evaluate(fitness);
            //print_best(0);

            for (std::size_t generation = 1; generation <= generations; generation++) {
                select();
                reproduce(generation);
                evaluate(fitness);

                //print_best(generation);
            }
        }

        template<typename PopulationEvaluator>
        void run_with(
            std::size_t generations,
            PopulationEvaluator& evaluator
        ) {
            evaluate_with(evaluator);
            update_global_best(0);
            print_best(0);

            for (std::size_t generation = 1; generation <= generations; generation++) {
                select();
                reproduce(generation);
                evaluate_with(evaluator);
                update_global_best(generation);
                print_best(generation);

            }

            print_global_best();
        }

        void update_global_best(std::size_t generation) {
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

            if (!has_global_best ||
                optimization.is_better(fitness_values[best_index], global_best_fitness)
            ) {
                global_best_fitness = fitness_values[best_index];
                global_best_generation = generation;
                has_global_best = true;

                global_best_chromosome.resize(chromosome_size);

                GA_CUDA_CHECK(cudaMemcpy(
                    global_best_chromosome.data(),
                    d_population + best_index * chromosome_size,
                    chromosome_size * sizeof(GeneType),
                    cudaMemcpyDeviceToHost
                ));
            }
        }

        void print_global_best() const {
            if (!has_global_best) {
                std::cout << "No global best available.\n";
                return;
            }

            std::cout << "Global best"
                    << " | generation = " << global_best_generation
                    << " | fitness = " << global_best_fitness
                    << "\n";
            
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
                    << "\n";

        
        }

        const std::vector<GeneType>& best_chromosome() const {
            if (!has_global_best) {
                throw std::runtime_error("No global best available.");
            }

            return global_best_chromosome;
        }

        double best_fitness() const {
            if (!has_global_best) {
                throw std::runtime_error("No global best available.");
            }

            return global_best_fitness;
        }

        std::size_t best_generation() const {
            if (!has_global_best) {
                throw std::runtime_error("No global best available.");
            }

            return global_best_generation;
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

        std::size_t population_size;
        std::size_t chromosome_size;
        std::size_t total_genes;
        std::size_t pair_count;

        Mutation mutation;
        Crossover crossover;
        Selection selection;
        Optimization optimization;

        GeneType* d_population = nullptr;
        GeneType* d_next_population = nullptr;

        double* d_fitness_values = nullptr;

        int* d_parent_a_indices = nullptr;
        int* d_parent_b_indices = nullptr;

        curandState* d_rng_states = nullptr;

        std::vector<GeneType> global_best_chromosome;
        double global_best_fitness = 0.0;
        bool has_global_best = false;
        std::size_t global_best_generation = 0;
    };

}