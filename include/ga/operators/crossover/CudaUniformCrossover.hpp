#pragma once

#include <curand_kernel.h>

namespace ga::crossover
{

    class CudaUniformCrossover
    {
        public:
            __host__ __device__
            CudaUniformCrossover() = default;

            template<typename GeneType>
            __device__
            void operator()(
                GeneType allele_a,
                GeneType allele_b,
                GeneType& child_a,
                GeneType& child_b,
                curandState& rng_state
            ) const
            {
                bool swap_alleles = (curand(&rng_state) & 1u) != 0u;

                if (swap_alleles) {
                    child_a = allele_b;
                    child_b = allele_a;

                } else {
                    child_a = allele_a;
                    child_b = allele_b;
                }
            }
    };

}