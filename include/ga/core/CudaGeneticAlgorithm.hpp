#pragma once

#include <cstddef>
#include <vector>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <algorithm>
#include <numeric>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "ga/utils/CudaCheck.cuh"
#include "ga/kernels/CudaKernels.cuh"
#include "ga/concepts/CudaMutationConcept.hpp"
#include "ga/concepts/CudaCrossoverConcept.hpp"
#include "ga/concepts/CudaSelectionConcept.hpp"
#include "ga/concepts/CudaFitnessComparatorConcept.hpp"

namespace ga::core {

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
        typename Mutation,
        typename Crossover,
        typename Selection,
        typename FitnessComparator
    >
    requires
        ga::concepts::CudaMutationConcept<Mutation, GeneType>                   &&
        ga::concepts::CudaCrossoverConcept<Crossover, GeneType>                 &&
        ga::concepts::CudaSelectionConcept<Selection, FitnessComparator>        && 
        ga::concepts::CudaFitnessComparatorConcept<FitnessComparator, GeneType>

    class CudaGeneticAlgorithm 
    {
        public:
            // -------------------------------------------------------------------------
            // Construction and lifetime
            // -------------------------------------------------------------------------

            /**
             * Creates a CUDA genetic algorithm instance and allocates all device buffers.
             *
             * @param population_size_ Number of individuals in the population.
             * @param chromosome_size_ Number of genes in each chromosome.
             * @param elitism_count_ Number of genes in each chromosome.
             * @param mutation_ Mutation operator used during reproduction.
             * @param crossover_ Crossover operator used during reproduction.
             * @param selection_ Selection policy used to choose parent pairs.
             * @param fitness_comparator_ Comparator used to rank fitness values.
             *
             * @throws std::invalid_argument If the population size is smaller than 2,
             * if the population size is odd, or if the chromosome size is zero.
             */
            CudaGeneticAlgorithm(
                std::size_t population_size_,
                std::size_t chromosome_size_,
                std::size_t elitism_count_,
                Mutation mutation_,
                Crossover crossover_,
                Selection selection_,
                FitnessComparator fitness_comparator_

            )
                : population_size(population_size_),
                  chromosome_size(chromosome_size_),
                  total_genes(population_size_ * chromosome_size_),
                  elitism_count(elitism_count_),
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

            /**
             * Initializes the population using a uniform distribution.
             *
             * Each gene is sampled independently from the interval
             * [min_value, max_value]. The CUDA random states are reset using the
             * provided seed before the population is generated.
             *
             * @param min_value Lower bound for each generated gene.
             * @param max_value Upper bound for each generated gene.
             * @param seed Seed used to initialize CUDA random states.
             */
            void create_uniform_population(
                float min_value,
                float max_value,
                unsigned long seed = 1234

            ) {
                ga::kernels::launch_setup_rng(
                    d_rng_states,
                    total_genes,
                    seed
                );

                ga::kernels::launch_create_uniform_population(
                    d_population,
                    d_rng_states,
                    total_genes,
                    min_value,
                    max_value
                );
            }

            /**
             * Initializes the population using a truncated Gaussian distribution.
             *
             * The underlying Gaussian distribution is centered at the middle of
             * the interval:
             *
             *     mean = (min_value + max_value) / 2
             *
             * The generated genes are constrained to [min_value, max_value].
             * The CUDA random states are reset using the provided seed before
             * the population is generated.
             *
             * @param min_value Lower bound for each generated gene.
             * @param max_value Upper bound for each generated gene.
             * @param sigma Standard deviation of the underlying Gaussian distribution.
             * @param seed Seed used to initialize CUDA random states.
             */
            void create_truncated_gaussian_population(
                float min_value,
                float max_value,
                float sigma,
                unsigned long seed = 1234

            ) {
                ga::kernels::launch_setup_rng(
                    d_rng_states,
                    total_genes,
                    seed
                );

                ga::kernels::launch_create_truncated_gaussian_population(
                    d_population,
                    d_rng_states,
                    total_genes,
                    min_value,
                    max_value,
                    sigma
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
                    d_population,
                    d_fitness_values,
                    population_size,
                    chromosome_size,
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
                    d_population,
                    d_fitness_values,
                    population_size,
                    chromosome_size
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
                    d_rng_states,
                    population_size,
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
            void reproduce(std::size_t generation)
            {
                auto elite_indices = find_elite_indices_host();

                const std::size_t offspring_count = population_size - elitism_count;

                ga::kernels::launch_crossover_mutation(
                    generation,
                    d_population,
                    d_next_population,
                    d_parent_a_indices,
                    d_parent_b_indices,
                    d_rng_states,
                    chromosome_size,
                    offspring_count,
                    crossover,
                    mutation
                );

                copy_elites_to_next_population(elite_indices);

                std::swap(d_population, d_next_population);
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
                update_global_best(0);
                print_best_fitness(0);

                for (std::size_t generation = 1; generation <= generations; generation++) {
                    select();
                    reproduce(generation);
                    evaluate(fitness);
                    update_global_best(generation);
                    print_best_fitness(generation);

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
                update_global_best(0);
                print_best_fitness(0);

                for (std::size_t generation = 1; generation <= generations; generation++) {
                    select();
                    reproduce(generation);
                    evaluate_with(evaluator);
                    update_global_best(generation);
                    print_best_fitness(generation);

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
            void update_global_best(std::size_t generation) 
            {
                auto fitness_values = copy_fitness_to_host();

                std::size_t best_index = 0;

                for (std::size_t i = 1; i < population_size; i++)
                {
                    if (fitness_comparator.is_better(fitness_values[i], fitness_values[best_index]))
                    {
                        best_index = i;

                    }

                }

                if (!has_global_best || fitness_comparator.is_better(fitness_values[best_index], global_best_fitness))
                {
                    has_global_best        = true;
                    global_best_fitness    = fitness_values[best_index];
                    global_best_generation = generation;

                    global_best_chromosome.resize(chromosome_size);

                    GA_CUDA_CHECK(cudaMemcpy(
                        global_best_chromosome.data(),
                        d_population + best_index * chromosome_size,
                        chromosome_size * sizeof(GeneType),
                        cudaMemcpyDeviceToHost
                    ));
                }
            }

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
            void print_best_fitness(std::size_t generation) const {
                auto fitness_values = copy_fitness_to_host();

                std::size_t best_index = 0;

                for (std::size_t i = 1; i < population_size; i++) {
                    if (fitness_comparator.is_better(
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

                for (std::size_t individual = 0; individual < population_size; individual++)
                {
                    std::cout << "[";

                    for (std::size_t allele = 0; allele < chromosome_size; allele++)
                    {
                        std::size_t index = individual * chromosome_size + allele;

                        std::cout << h_population[index];

                        if (allele < chromosome_size - 1)
                        {
                            std::cout << ", ";
                        }
                    }

                    std::cout << "]" << std::endl;
                }
            }

        private:
            std::size_t population_size;
            std::size_t chromosome_size;
            std::size_t total_genes;
            std::size_t elitism_count;

            Mutation mutation;
            Crossover crossover;
            Selection selection;
            FitnessComparator fitness_comparator;

            GeneType* d_population      = nullptr;
            GeneType* d_next_population = nullptr;

            double* d_fitness_values = nullptr;

            int* d_parent_a_indices = nullptr;
            int* d_parent_b_indices = nullptr;

            curandState* d_rng_states = nullptr;

            std::vector<GeneType> global_best_chromosome;
            double global_best_fitness         = 0.0;
            bool has_global_best               = false;
            std::size_t global_best_generation = 0;

            void validate()
            {
                if (elitism_count >= population_size) {
                    throw std::invalid_argument(
                        "Elitism count must be smaller than population size."
                    );
                }

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
            }

            void allocate() 
            {
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
                    (population_size / 2) * sizeof(int)
                ));

                GA_CUDA_CHECK(cudaMalloc(
                    &d_parent_b_indices,
                    (population_size / 2) * sizeof(int)
                ));

                GA_CUDA_CHECK(cudaMalloc(
                    &d_rng_states,
                    total_genes * sizeof(curandState)
                ));
            }

            void release() 
            {
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

            std::vector<std::size_t> find_elite_indices_host() const
            {
                std::vector<std::size_t> elite_indices;

                if (elitism_count == 0) {
                    return elite_indices;
                }

                auto fitness_values = copy_fitness_to_host();

                std::vector<std::size_t> indices(population_size);

                std::iota(
                    indices.begin(),
                    indices.end(),
                    static_cast<std::size_t>(0)
                );

                std::partial_sort(
                    indices.begin(),
                    indices.begin() + elitism_count,
                    indices.end(),
                    [this, &fitness_values](std::size_t a, std::size_t b) {
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

            void copy_elites_to_next_population(
                const std::vector<std::size_t>& elite_indices
            )
            {
                for (std::size_t elite_position = 0; elite_position < elite_indices.size(); elite_position++)
                {
                    std::size_t source_index = elite_indices[elite_position];

                    const GeneType* source = d_population + source_index * chromosome_size;

                    GeneType* destination = d_next_population + (elite_position + (population_size - elitism_count)) * chromosome_size;

                    GA_CUDA_CHECK(cudaMemcpy(
                        destination,
                        source,
                        chromosome_size * sizeof(GeneType),
                        cudaMemcpyDeviceToDevice
                    ));
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
                std::vector<GeneType> h_population(total_genes);

                GA_CUDA_CHECK(cudaMemcpy(
                    h_population.data(),
                    d_population,
                    total_genes * sizeof(GeneType),
                    cudaMemcpyDeviceToHost
                ));

                return h_population;
            }
    };

}