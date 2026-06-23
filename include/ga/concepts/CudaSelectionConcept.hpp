#pragma once

#include <concepts>
#include <cstddef>
#include <curand_kernel.h>

namespace ga::concepts {

    template<typename Selection, typename FitnessComparator>
    concept CudaSelectionConcept =
        requires(
            Selection selection,
            const double* fitness_values,
            std::size_t population_size,
            FitnessComparator fitness_comparator,
            curandState& rng_state
        ){
            {
                selection(
                    fitness_values,
                    population_size,
                    fitness_comparator,
                    rng_state
                )
            } -> std::convertible_to<int>;
        };

}