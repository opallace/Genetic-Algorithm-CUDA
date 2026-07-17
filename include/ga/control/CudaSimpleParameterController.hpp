#pragma once

#include <stdexcept>
#include <algorithm>
#include <cstddef>
#include <cmath>
#include <iostream>

#include "ga/concepts/CudaFitnessComparatorConcept.hpp"

namespace ga::control
{

    class CudaSimpleParameterController
    {
        public:

            std::size_t stagnation_threshold        = 10;
            std::size_t strong_stagnation_threshold = 30;
            float strong_improvement_threshold      = 0.05f;
            double improvement_epsilon              = 1.0e-4;

            float min_mutation_rate = 0.001f;
            float max_mutation_rate = 0.05f;

            float min_mutation_eta = 5.0f;
            float max_mutation_eta = 80.0f;

            float min_crossover_eta = 5.0f;
            float max_crossover_eta = 50.0f;

            std::size_t min_population_size = 32;
            std::size_t max_population_size = 128;

            float improvement_mutation_rate_factor      = 0.98f;
            float improvement_mutation_eta_factor       = 1.02f;
            float improvement_crossover_eta_factor      = 1.01f;
            std::ptrdiff_t improvement_population_delta = 0;

            float strong_improvement_mutation_rate_factor      = 0.90f;
            float strong_improvement_mutation_eta_factor       = 1.10f;
            float strong_improvement_crossover_eta_factor      = 1.05f;
            std::ptrdiff_t strong_improvement_population_delta = -16;

            float stagnation_mutation_rate_factor      = 1.25f;
            float stagnation_mutation_eta_factor       = 0.85f;
            float stagnation_crossover_eta_factor      = 0.90f;
            std::ptrdiff_t stagnation_population_delta = 16;

            float strong_stagnation_mutation_rate_factor      = 1.50f;
            float strong_stagnation_mutation_eta_factor       = 0.70f;
            float strong_stagnation_crossover_eta_factor      = 0.80f;
            std::ptrdiff_t strong_stagnation_population_delta = 32;

            template<
                typename Mutation,
                typename Crossover,
                typename Selection,
                typename FitnessComparator,
                typename Population,
                typename PopulationInitializer
            >
            void update(
                double previous_best_fitness,
                double current_best_fitness,

                Mutation& mutation,
                Crossover& crossover,
                Selection& selection,
                FitnessComparator& fitness_comparator,

                Population& population,
                PopulationInitializer initializer,
                std::size_t elitism_count,
                unsigned long seed
            )
            {

                const double improvement_magnitude =
                    std::abs(current_best_fitness - previous_best_fitness);

                const bool improved =
                    fitness_comparator.is_better(
                        current_best_fitness,
                        previous_best_fitness
                    ) &&
                    improvement_magnitude > improvement_epsilon;

                if (improved)
                {
                    stagnation_count = 0;

                    if (improvement_magnitude >= strong_improvement_threshold)
                    {
                        mutation.mutation_rate *= strong_improvement_mutation_rate_factor;
                        mutation.eta           *= strong_improvement_mutation_eta_factor;
                        crossover.eta          *= strong_improvement_crossover_eta_factor;

                        apply_population_delta(
                            strong_improvement_population_delta,
                            population,
                            initializer,
                            elitism_count,
                            seed
                        );
                    }
                    else
                    {
                        mutation.mutation_rate *= improvement_mutation_rate_factor;
                        mutation.eta           *= improvement_mutation_eta_factor;
                        crossover.eta          *= improvement_crossover_eta_factor;

                        apply_population_delta(
                            improvement_population_delta,
                            population,
                            initializer,
                            elitism_count,
                            seed
                        );
                    }
                }
                else
                {
                    stagnation_count++;

                    if (stagnation_count % strong_stagnation_threshold == 0)
                    {
                        mutation.mutation_rate *= strong_stagnation_mutation_rate_factor;
                        mutation.eta           *= strong_stagnation_mutation_eta_factor;
                        crossover.eta          *= strong_stagnation_crossover_eta_factor;

                        apply_population_delta(
                            strong_stagnation_population_delta,
                            population,
                            initializer,
                            elitism_count,
                            seed
                        );
                    }
                    else if (stagnation_count % stagnation_threshold == 0)
                    {
                        mutation.mutation_rate *= stagnation_mutation_rate_factor;
                        mutation.eta           *= stagnation_mutation_eta_factor;
                        crossover.eta          *= stagnation_crossover_eta_factor;

                        apply_population_delta(
                            stagnation_population_delta,
                            population,
                            initializer,
                            elitism_count,
                            seed
                        );
                    }
                }

                mutation.mutation_rate =
                    std::clamp(
                        mutation.mutation_rate,
                        min_mutation_rate,
                        max_mutation_rate
                    );

                mutation.eta =
                    std::clamp(
                        mutation.eta,
                        min_mutation_eta,
                        max_mutation_eta
                    );

                crossover.eta =
                    std::clamp(
                        crossover.eta,
                        min_crossover_eta,
                        max_crossover_eta
                    );
            }

        private:

            std::size_t stagnation_count = 0;
        
            template<
                typename Population,
                typename PopulationInitializer
            >
            void apply_population_delta(
                std::ptrdiff_t delta,
                Population& population,
                PopulationInitializer initializer,
                std::size_t elitism_count,
                unsigned long seed
            ) const
            {
                if (delta == 0){
                    return;
                }

                const std::ptrdiff_t current_size = static_cast<std::ptrdiff_t>(population.size());

                std::size_t effective_min_size = std::max(min_population_size, elitism_count + 2);

                if (effective_min_size % 2 != 0){
                    effective_min_size++;
                }

                std::size_t effective_max_size = std::min(max_population_size, population.capacity());

                if (effective_max_size % 2 != 0){
                    effective_max_size--;
                }

                if (effective_min_size > effective_max_size){
                    throw std::invalid_argument(
                        "Invalid population bounds after elitism protection."
                    );
                }

                std::ptrdiff_t requested_size = current_size + delta;

                requested_size =
                    std::clamp(
                        requested_size,
                        static_cast<std::ptrdiff_t>(effective_min_size),
                        static_cast<std::ptrdiff_t>(effective_max_size)
                    );

                if (requested_size % 2 != 0){
                    if (delta > 0){
                        requested_size++;

                    }else {
                        requested_size--;

                    }
                }

                requested_size =
                    std::clamp(
                        requested_size,
                        static_cast<std::ptrdiff_t>(effective_min_size),
                        static_cast<std::ptrdiff_t>(effective_max_size)
                    );

                if (requested_size == current_size){
                    return;
                }

                population.resize(
                    static_cast<std::size_t>(requested_size),
                    initializer,
                    elitism_count,
                    seed
                );

                std::cout
                    << "Population resized"
                    << " | from = " << current_size
                    << " | to = " << requested_size
                    << "\n";
            }
    };

}