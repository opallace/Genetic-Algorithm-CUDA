#pragma once

#include <cstddef>
#include <curand_kernel.h>

namespace ga::selection {

    /**
     * @brief Tournament selection operator for CUDA genetic algorithms.
     *
     * This operator selects one parent by randomly sampling a fixed number of
     * candidate individuals and returning the best candidate according to the
     * provided fitness comparator.
     */
    class CudaTournamentSelection {
        public:
            /**
             * @brief Number of candidates sampled in each tournament.
             *
             * Larger tournament sizes increase selection pressure.
             */
            std::size_t tournament_size;

            /**
             * @brief Constructs a tournament selection operator.
             *
             * @param tournament_size_ Number of individuals sampled per tournament.
             */
            __host__ __device__
            explicit CudaTournamentSelection(std::size_t tournament_size_)
                : tournament_size(tournament_size_)
            {}

            /**
             * @brief Selects one parent index using tournament selection.
             *
             * @tparam FitnessComparator Comparator used to decide whether one
             * fitness value is better than another.
             *
             * @param fitness_values Device array containing one fitness value per individual.
             * @param population_size Number of individuals in the population.
             * @param fitness_comparator Fitness comparator used for maximization or minimization.
             * @param rng_state CURAND random state.
             *
             * @return Index of the selected parent.
             */
            template<typename FitnessComparator>
            __device__
            int operator()(
                const double* fitness_values,
                std::size_t population_size,
                FitnessComparator fitness_comparator,
                curandState& rng_state
            ) const {
                int best_index = static_cast<int>(
                    curand(&rng_state) % population_size
                );

                double best_fitness = fitness_values[best_index];

                for (std::size_t i = 1; i < tournament_size; i++) 
                {
                    int candidate_index = static_cast<int>(
                        curand(&rng_state) % population_size
                    );

                    double candidate_fitness = fitness_values[candidate_index];

                    if (fitness_comparator.is_better(candidate_fitness, best_fitness)) 
                    {
                        best_index   = candidate_index;
                        best_fitness = candidate_fitness;
                    }
                }

                return best_index;
            }
    };

}