#pragma once

#include <curand_kernel.h>

namespace ga::crossover {

    class CudaUniformCrossover {
        public:

            template<typename GeneType>
            __device__
            GeneType operator()(GeneType allele_a, GeneType allele_b, curandState& rng_state) {
                bool choose_a = (curand(&rng_state) % 2) == 0;
                return choose_a ? allele_a : allele_b;
            }

    };

}