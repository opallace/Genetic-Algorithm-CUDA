#pragma once

#include <cstddef>

#include "problem/SuspendedLoadProblem.hpp"

namespace suspended_load::solver {

    void launch_evaluate_suspended_load_fitness(
        const float* d_population,
        double* d_fitness_values,
        std::size_t population_size,
        std::size_t chromosome_size,
        suspended_load::problem::SuspendedLoadProblem problem,
        suspended_load::problem::SuspendedLoadFitnessWeights weights
    );

} // namespace suspended_load::solver
