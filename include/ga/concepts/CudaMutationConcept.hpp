#pragma once

#include <concepts>
#include <curand_kernel.h>

namespace ga::cuda::concepts {

    template<typename Mutation, typename GeneType>
    concept CudaMutationConcept =
        requires(Mutation mutation, GeneType allele, curandState& rng_state){
            { mutation(allele, rng_state) } -> std::convertible_to<GeneType>;
            
        };

}