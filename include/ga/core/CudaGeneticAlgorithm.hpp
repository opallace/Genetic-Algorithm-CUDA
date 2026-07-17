#pragma once

#include <cstddef>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <numeric>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "ga/utils/CudaCheck.cuh"
#include "ga/utils/GATelemetry.hpp"

#include "ga/concepts/CudaMutationConcept.hpp"
#include "ga/concepts/CudaCrossoverConcept.hpp"
#include "ga/concepts/CudaSelectionConcept.hpp"
#include "ga/concepts/CudaFitnessComparatorConcept.hpp"
#include "ga/concepts/CudaPopulationInitializerConcept.hpp"

#include "ga/kernels/CudaEvaluationKernels.cuh"
#include "ga/kernels/CudaRandomKernels.cuh"
#include "ga/kernels/CudaReproductionKernels.cuh"
#include "ga/kernels/CudaSelectionKernels.cuh"

namespace ga::core {

    struct GenerationStats{
        std::size_t best_index = 0;
        double best_fitness    = 0.0;
        std::vector<std::size_t> elite_indices;

    };

    /**
     * CUDA-based genetic algorithm.
     *
     * This class stores the population, fitness values, parent indices, and
     * random number generator states in CUDA device memory. The genetic
     * operators are provided as template parameters and are executed by CUDA
     * kernels through the launch functions defined in ga::kernels.
     *
     * The current implementation assumes an even population size, because the
     * selection step creates parent pairs and the reproduction step creates the
     * next population from those pairs.
     *
     * @tparam GeneType Type used to store each gene in the chromosome.
     * @tparam Mutation Device-callable mutation operator.
     * @tparam Crossover Device-callable crossover operator.
     * @tparam Selection Selection policy used to choose parent pairs.
     * @tparam FitnessComparator Comparator used to decide whether one fitness
     * value is better than another.
     */
    template<
        typename GeneType,
        typename Population,
        typename Mutation,
        typename Crossover,
        typename Selection,
        typename FitnessComparator
    >
    requires
        ga::concepts::CudaMutationConcept<Mutation, GeneType>                   &&
        ga::concepts::CudaCrossoverConcept<Crossover, GeneType>                 &&
        ga::concepts::CudaSelectionConcept<Selection, FitnessComparator>        && 
        ga::concepts::CudaFitnessComparatorConcept<FitnessComparator, double>

    class CudaGeneticAlgorithm 
    {
        public:
            
            CudaGeneticAlgorithm(
                std::size_t elitism_count_,
                float crossover_rate_,

                Population& population_,
                Mutation mutation_,
                Crossover crossover_,
                Selection selection_,
                FitnessComparator fitness_comparator_
            )
                : elitism_count(elitism_count_),
                  crossover_rate(crossover_rate_), 

                  population(population_),
                  mutation(mutation_),
                  crossover(crossover_),
                  selection(selection_),
                  fitness_comparator(fitness_comparator_)
            {
                validate();
                allocate();
            }

            /**
             * Releases all CUDA device buffers owned by the genetic algorithm.
             */
            ~CudaGeneticAlgorithm() {
                release();
            }

            /**
             * Copy construction is disabled because this class owns CUDA device memory.
             */
            CudaGeneticAlgorithm(const CudaGeneticAlgorithm&) = delete;

            /**
             * Copy assignment is disabled because this class owns CUDA device memory.
             */
            CudaGeneticAlgorithm& operator=(const CudaGeneticAlgorithm&) = delete;

            // -------------------------------------------------------------------------
            // Population initialization
            // -------------------------------------------------------------------------

            template<typename PopulationInitializer>
            requires ga::concepts::CudaPopulationInitializerConcept<
                PopulationInitializer,
                GeneType
            >
            void create_population(
                PopulationInitializer initializer,
                unsigned long seed = 1234
            )
            {
                population.initialize(initializer, seed);

                ga::kernels::launch_setup_rng(
                    d_pair_rng_states,
                    population.capacity() / 2,
                    seed + 1337
                );
            }

            // -------------------------------------------------------------------------
            // Fitness evaluation
            // -------------------------------------------------------------------------

            /**
             * Evaluates the fitness of every individual using a device-callable functor.
             *
             * The fitness functor is launched through a CUDA kernel and receives a
             * CudaChromosome view for each individual.
             *
             * @tparam Fitness Type of the device-callable fitness functor.
             * @param fitness Fitness functor used to evaluate one chromosome.
             */
            template<typename Fitness>
            void evaluate(Fitness fitness) {
                ga::kernels::launch_evaluate_fitness(
                    population.data(),
                    d_fitness_values,
                    population.size(),
                    population.chromosome_length(),
                    fitness
                );
            }

            /**
             * Evaluates the population using an external population evaluator.
             *
             * This overload is useful when the fitness computation needs a custom
             * evaluator that manages additional CUDA buffers, libraries, or batched
             * execution strategies.
             *
             * @tparam PopulationEvaluator Type of the external evaluator.
             * @param evaluator Evaluator object that writes fitness values to device memory.
             */
            template<typename PopulationEvaluator>
            void evaluate_with(PopulationEvaluator& evaluator) {
                evaluator.evaluate_population(
                    population.data(),
                    d_fitness_values,
                    population.size(),
                    population.chromosome_length()
                );
            }

            // -------------------------------------------------------------------------
            // Selection
            // -------------------------------------------------------------------------

            /**
             * Selects parent pairs for the next reproduction step.
             *
             * This method assumes that the current population has already been
             * evaluated and that d_fitness_values contains valid fitness values.
             */
            void select() {
                ga::kernels::launch_selection(
                    d_fitness_values,
                    d_parent_a_indices,
                    d_parent_b_indices,
                    d_pair_crosses,
                    d_pair_rng_states,
                    population.size(),
                    crossover_rate,
                    selection,
                    fitness_comparator
                );
            }

            // -------------------------------------------------------------------------
            // Reproduction: crossover and mutation
            // -------------------------------------------------------------------------

            /**
             * Creates the next population through crossover and mutation.
             *
             * This method assumes that select() has already filled the parent index
             * buffers. After the reproduction kernel finishes, the current population
             * buffer and the next population buffer are swapped.
             *
             * @param generation Current generation index.
             */
            void reproduce(
                const std::vector<std::size_t>& elite_indices
            )
            {
                if (elite_indices.size() != elitism_count)
                {
                    throw std::runtime_error(
                        "Elite index count does not match elitism_count."
                    );
                }

                const std::size_t offspring_count = population.size() - elitism_count;

                ga::kernels::launch_crossover_mutation(
                    population.data(),
                    population.next_data(),
                    d_parent_a_indices,
                    d_parent_b_indices,
                    d_pair_crosses,
                    population.gene_rng_states(),
                    population.chromosome_length(),
                    offspring_count,
                    crossover,
                    mutation
                );

                copy_elites_to_next_population(elite_indices);

                population.swap_buffers();
            }

            // -------------------------------------------------------------------------
            // Evolution loop
            // -------------------------------------------------------------------------

            /**
             * Runs the genetic algorithm for a fixed number of generations.
             *
             * This overload evaluates fitness using a device-callable functor.
             * In the current implementation, this method does not update the
             * global-best tracking fields.
             *
             * @tparam Fitness Type of the device-callable fitness functor.
             * @param generations Number of generations to execute.
             * @param fitness Fitness functor used to evaluate one chromosome.
             */
            template<typename Fitness>
            void run(std::size_t generations, Fitness fitness) {
                evaluate(fitness);
                GenerationStats stats = compute_generation_stats();
                update_global_best(0, stats);
                print_best_fitness(0, stats);

                for (std::size_t generation = 1; generation <= generations; generation++) {
                    select();
                    reproduce(stats.elite_indices);
                    evaluate(fitness);
                    stats = compute_generation_stats();
                    update_global_best(generation, stats);
                    print_best_fitness(generation, stats);

                }

                print_global_best_fitness();
            }

            /**
             * Runs the genetic algorithm using an external population evaluator.
             *
             * This overload updates and prints the global best solution after each
             * generation.
             *
             * @tparam PopulationEvaluator Type of the external evaluator.
             * @param generations Number of generations to execute.
             * @param evaluator Evaluator object that writes fitness values to device memory.
             */
            template<typename PopulationEvaluator>
            void run_with(
                std::size_t generations,
                PopulationEvaluator& evaluator
            ) {
                evaluate_with(evaluator);
                GenerationStats stats = compute_generation_stats();
                update_global_best(0, stats);
                print_best_fitness(0, stats);

                for (std::size_t generation = 1; generation <= generations; generation++) {
                    select();
                    reproduce(stats.elite_indices);
                    evaluate_with(evaluator);
                    stats = compute_generation_stats();
                    update_global_best(generation, stats);
                    print_best_fitness(generation, stats);

                }

                print_global_best_fitness();
            }

            template<
                typename PopulationEvaluator,
                typename ParameterController,
                typename PopulationInitializer
            >
            requires ga::concepts::CudaPopulationInitializerConcept<
                PopulationInitializer,
                GeneType
            >
            void run_with(
                std::size_t generations,
                PopulationEvaluator& evaluator,
                ParameterController& parameter_controller,
                PopulationInitializer population_initializer,
                unsigned long seed = 1234

            )
            {
                evaluate_with(evaluator);
                GenerationStats stats = compute_generation_stats();
                update_global_best(0, stats);
                print_best_fitness(0, stats);

                double last_previous_best_fitness = global_best_fitness;
                double last_current_best_fitness  = global_best_fitness;

                plotter.add_record(
                    0,
                    stats.best_fitness,
                    global_best_fitness,
                    mutation.mutation_rate,
                    mutation.eta,
                    crossover.eta,
                    population.size()
                );

                for (std::size_t generation = 1; generation <= generations; generation++){
                    select();
                    reproduce(stats.elite_indices);

                    parameter_controller.update(
                        last_previous_best_fitness,
                        last_current_best_fitness,

                        mutation,
                        crossover,
                        selection,
                        fitness_comparator,

                        population,
                        population_initializer,
                        elitism_count,
                        seed + generation
                    );

                    evaluate_with(evaluator);

                    const double previous_best_fitness = global_best_fitness;

                    stats = compute_generation_stats();
                    update_global_best(generation, stats);
                    print_best_fitness(generation, stats);

                    last_previous_best_fitness = previous_best_fitness;
                    last_current_best_fitness  = global_best_fitness;

                    plotter.add_record(
                        generation,
                        stats.best_fitness,
                        global_best_fitness,
                        mutation.mutation_rate,
                        mutation.eta,
                        crossover.eta,
                        population.size()
                    );
                }

                print_global_best_fitness();
            }

            // -------------------------------------------------------------------------
            // Best-solution tracking
            // -------------------------------------------------------------------------

            /**
             * Updates the best chromosome found across all evaluated generations.
             *
             * The method copies fitness values to host memory, finds the best
             * individual according to the configured fitness comparator, and stores
             * the corresponding chromosome on the host if it improves the global best.
             *
             * @param generation Generation associated with the current population.
             */

            /**
             * Prints the best solution found across all evaluated generations.
             */
            void print_global_best_fitness() const {
                if (!has_global_best) {
                    std::cout << "No global best available.\n";
                    return;
                }

                std::cout << "Global best"
                        << " | generation = " << global_best_generation
                        << " | fitness = " << global_best_fitness
                        << "\n";
                
            }

            /**
             * Prints the best fitness value in the current population.
             *
             * @param generation Generation index shown in the printed message.
             */
            void print_best_fitness(
                std::size_t generation,
                const GenerationStats& stats
            ) const
            {
                std::cout
                    << "Generation "
                    << generation
                    << " | best fitness = "
                    << stats.best_fitness
                    << "\n";
            }

            /**
             * Returns the best chromosome found across all evaluated generations.
             *
             * @return Host vector containing the best chromosome.
             *
             * @throws std::runtime_error If no global best solution is available.
             */
            const std::vector<GeneType>& best_chromosome() const {
                if (!has_global_best) {
                    throw std::runtime_error("No global best available.");
                }

                return global_best_chromosome;
            }

            /**
             * Returns the best fitness found across all evaluated generations.
             *
             * @return Best tracked fitness value.
             *
             * @throws std::runtime_error If no global best solution is available.
             */
            double best_fitness() const {
                if (!has_global_best) {
                    throw std::runtime_error("No global best available.");
                }

                return global_best_fitness;
            }

            /**
             * Returns the generation where the global best solution was found.
             *
             * @return Generation index associated with the best tracked solution.
             *
             * @throws std::runtime_error If no global best solution is available.
             */
            std::size_t best_generation() const {
                if (!has_global_best) {
                    throw std::runtime_error("No global best available.");
                }

                return global_best_generation;
            }

            void print_population() const
            {
                auto h_population = copy_population_to_host();

                for (std::size_t individual = 0; individual < population.size(); individual++)
                {
                    std::cout << "[";

                    for (std::size_t allele = 0; allele < population.chromosome_length(); allele++)
                    {
                        std::size_t index = individual * population.chromosome_length() + allele;

                        std::cout << h_population[index];

                        if (allele < population.chromosome_length() - 1)
                        {
                            std::cout << ", ";
                        }
                    }

                    std::cout << "]" << std::endl;
                }
            }

        private:
            Population& population;
            ga::utils::GATelemetry plotter;
            
            std::size_t elitism_count;
            float crossover_rate;

            Mutation mutation;
            Crossover crossover;
            Selection selection;
            FitnessComparator fitness_comparator;

            double* d_fitness_values = nullptr;

            int* d_parent_a_indices = nullptr;
            int* d_parent_b_indices = nullptr;
            unsigned char* d_pair_crosses = nullptr;

            curandState* d_pair_rng_states = nullptr;

            std::vector<GeneType> global_best_chromosome;
            double global_best_fitness          = 0.0;
            bool has_global_best                = false;
            std::size_t global_best_generation  = 0;

            void validate()
            {
                if (crossover_rate < 0.0f || crossover_rate > 1.0f) {
                    throw std::invalid_argument(
                        "Crossover rate must be in [0, 1]."
                    );
                }

                if (elitism_count >= population.size()) {
                    throw std::invalid_argument(
                        "Elitism count must be smaller than population size."
                    );
                }

                if (elitism_count % 2 != 0) {
                    throw std::invalid_argument(
                        "CUDA GA currently requires even elitism count."
                    );
                }

                if (population.size() < 2) {
                    throw std::invalid_argument(
                        "Population size must be at least 2."
                    );
                }

                if (population.size() % 2 != 0) {
                    throw std::invalid_argument(
                        "CUDA GA currently requires even population size."
                    );
                }

                if (population.chromosome_length() == 0) {
                    throw std::invalid_argument(
                        "Chromosome size must be greater than 0."
                    );
                }
            }

            void allocate() 
            {
                GA_CUDA_CHECK(cudaMalloc(
                    &d_fitness_values,
                    population.capacity() * sizeof(double)
                ));

                GA_CUDA_CHECK(cudaMalloc(
                    &d_parent_a_indices,
                    (population.capacity() / 2) * sizeof(int)
                ));

                GA_CUDA_CHECK(cudaMalloc(
                    &d_parent_b_indices,
                    (population.capacity() / 2) * sizeof(int)
                ));

                GA_CUDA_CHECK(cudaMalloc(
                    &d_pair_crosses,
                    (population.capacity() / 2) * sizeof(unsigned char)
                ));

                GA_CUDA_CHECK(cudaMalloc(
                    &d_pair_rng_states,
                    (population.capacity() / 2) * sizeof(curandState)
                ));
            }

            void release() 
            {
                if (d_fitness_values) {
                    cudaFree(d_fitness_values);
                }

                if (d_parent_a_indices) {
                    cudaFree(d_parent_a_indices);
                }

                if (d_parent_b_indices) {
                    cudaFree(d_parent_b_indices);
                }

                if (d_pair_crosses) {
                    cudaFree(d_pair_crosses);
                }

                if (d_pair_rng_states) {
                    cudaFree(d_pair_rng_states);
                }
            }

            void copy_elites_to_next_population(
                const std::vector<std::size_t>& elite_indices
            )
            {
                for (std::size_t elite_position = 0; elite_position < elite_indices.size(); elite_position++)
                {
                    std::size_t source_index = elite_indices[elite_position];

                    const GeneType* source = population.data() + source_index * population.chromosome_length();

                    GeneType* destination = population.next_data() + (elite_position + (population.size() - elitism_count)) * population.chromosome_length();

                    GA_CUDA_CHECK(cudaMemcpy(
                        destination,
                        source,
                        population.chromosome_length() * sizeof(GeneType),
                        cudaMemcpyDeviceToDevice
                    ));
                }
            }

            std::vector<double> copy_fitness_to_host() const {
                std::vector<double> host_fitness(population.size());

                GA_CUDA_CHECK(cudaMemcpy(
                    host_fitness.data(),
                    d_fitness_values,
                    population.size() * sizeof(double),
                    cudaMemcpyDeviceToHost
                ));

                return host_fitness;
            }

            std::vector<GeneType> copy_population_to_host() const
            {
                return population.copy_to_host();
            }

            GenerationStats compute_generation_stats() const
            {
                const std::vector<double> fitness_values = copy_fitness_to_host();

                GenerationStats stats;

                stats.best_index    = find_best_index_from_fitness(fitness_values);
                stats.best_fitness  = fitness_values[stats.best_index];
                stats.elite_indices = find_elite_indices_from_fitness(fitness_values);

                return stats;
            }


            std::size_t find_best_index_from_fitness(
                const std::vector<double>& fitness_values
            ) const
            {
                std::size_t best_index = 0;

                for (std::size_t i = 1; i < population.size(); i++){
                    if (fitness_comparator.is_better(fitness_values[i], fitness_values[best_index] )){
                        best_index = i;
                    }
                }

                return best_index;
            }


            std::vector<std::size_t> find_elite_indices_from_fitness(
                const std::vector<double>& fitness_values
            ) const
            {
                std::vector<std::size_t> elite_indices;

                if (elitism_count == 0){
                    return elite_indices;
                }

                std::vector<std::size_t> indices(population.size());

                std::iota(
                    indices.begin(),
                    indices.end(),
                    static_cast<std::size_t>(0)
                );

                std::partial_sort(
                    indices.begin(),
                    indices.begin() + elitism_count,
                    indices.end(),
                    [this, &fitness_values](
                        std::size_t a,
                        std::size_t b
                    )
                    {
                        return fitness_comparator.is_better(
                            fitness_values[a],
                            fitness_values[b]
                        );
                    }
                );

                elite_indices.assign(
                    indices.begin(),
                    indices.begin() + elitism_count
                );

                return elite_indices;
            }


            void update_global_best(
                std::size_t generation,
                const GenerationStats& stats
            )
            {
                const bool improved =
                    !has_global_best ||
                    fitness_comparator.is_better(
                        stats.best_fitness,
                        global_best_fitness
                    );

                if (!improved)
                {
                    return;
                }

                has_global_best        = true;
                global_best_fitness    = stats.best_fitness;
                global_best_generation = generation;

                global_best_chromosome.resize(
                    population.chromosome_length()
                );

                GA_CUDA_CHECK(cudaMemcpy(
                    global_best_chromosome.data(),
                    population.data()
                        + stats.best_index * population.chromosome_length(),
                    population.chromosome_length() * sizeof(GeneType),
                    cudaMemcpyDeviceToHost
                ));
            }


            
    };

}