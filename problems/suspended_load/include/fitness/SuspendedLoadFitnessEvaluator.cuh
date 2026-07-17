#pragma once

#include <cstddef>
#include <stdexcept>

#include "problem/SuspendedLoadProblem.hpp"
#include "solver/SuspendedLoadRk4Solver.cuh"

namespace suspended_load::fitness {

    class SuspendedLoadFitnessEvaluator {
        public:
            SuspendedLoadFitnessEvaluator(
                suspended_load::problem::SuspendedLoadProblem problem_,
                suspended_load::problem::SuspendedLoadFitnessWeights weights_
            )
                : problem(problem_),
                  weights(weights_)
            {}

            void evaluate_population(
                const float* d_population,
                double* d_fitness_values,
                std::size_t population_size,
                std::size_t chromosome_size
            ) {
                if (!d_population) {
                    throw std::invalid_argument(
                        "SuspendedLoadFitnessEvaluator: d_population is null."
                    );
                }

                if (!d_fitness_values) {
                    throw std::invalid_argument(
                        "SuspendedLoadFitnessEvaluator: d_fitness_values is null."
                    );
                }

                if (population_size == 0) {
                    throw std::invalid_argument(
                        "SuspendedLoadFitnessEvaluator: population_size must be greater than zero."
                    );
                }

                if (chromosome_size == 0) {
                    throw std::invalid_argument(
                        "SuspendedLoadFitnessEvaluator: chromosome_size must be greater than zero."
                    );
                }

                if (problem.length <= 0.0f) {
                    throw std::invalid_argument(
                        "SuspendedLoadFitnessEvaluator: pendulum length must be positive."
                    );
                }

                if (problem.final_time <= 0.0f) {
                    throw std::invalid_argument(
                        "SuspendedLoadFitnessEvaluator: final_time must be positive."
                    );
                }

                if (problem.steps_per_gene <= 0) {
                    throw std::invalid_argument(
                        "SuspendedLoadFitnessEvaluator: steps_per_gene must be positive."
                    );
                }

                suspended_load::solver::launch_evaluate_suspended_load_fitness(
                    d_population,
                    d_fitness_values,
                    population_size,
                    chromosome_size,
                    problem,
                    weights
                );
            }

        private:
            suspended_load::problem::SuspendedLoadProblem problem;
            suspended_load::problem::SuspendedLoadFitnessWeights weights;
    };

} // namespace suspended_load::fitness
