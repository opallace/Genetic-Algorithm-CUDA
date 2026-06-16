#pragma once

#include <concepts>
#include <curand_kernel.h>

namespace ga::concepts {

    template<typename Mutation, typename GeneType>
    concept CudaMutationConcept =
        requires(Mutation mutation, std::size_t generation, GeneType allele, curandState& rng_state){
            { mutation(generation, allele, rng_state) } -> std::convertible_to<GeneType>;
            
        };

}