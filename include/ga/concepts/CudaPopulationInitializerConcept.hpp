#pragma once

#include <concepts>
#include <curand_kernel.h>

namespace ga::concepts {

    template<typename PopulationInitializer, typename GeneType>
    concept CudaPopulationInitializerConcept =
        requires(
            const PopulationInitializer initializer,
            curandState& rng_state
        ) {
            {
                initializer.template generate<GeneType>(rng_state)
            } -> std::convertible_to<GeneType>;
        };

}