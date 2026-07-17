#pragma once

#include <concepts>
#include <curand_kernel.h>

namespace ga::concepts
{

    template<typename Crossover, typename GeneType>
    concept CudaCrossoverConcept =
        requires(
            Crossover crossover,
            GeneType allele_a,
            GeneType allele_b,
            GeneType& child_a,
            GeneType& child_b,
            curandState& rng_state
        )
        {
            {
                crossover(
                    allele_a,
                    allele_b,
                    child_a,
                    child_b,
                    rng_state
                )
            } -> std::same_as<void>;
        };

}