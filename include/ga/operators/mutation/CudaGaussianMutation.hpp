#pragma once

#include <cstddef>
#include <cmath>

#include <curand_kernel.h>

namespace ga::mutation {

    class CudaGaussianMutation {
        public:
            float initial_sigma;
            float min_sigma;
            float decay_rate;

            __host__ __device__
            explicit CudaGaussianMutation(
                float initial_sigma,
                float min_sigma = 0.0001f,
                float decay_rate = 0.001f
            )
                : initial_sigma(initial_sigma),
                min_sigma(min_sigma),
                decay_rate(decay_rate)
            {}

            __host__ __device__
            float current_sigma(std::size_t generation) const {
                float g = static_cast<float>(generation);

                return min_sigma
                    + (initial_sigma - min_sigma)
                    * expf(-decay_rate * g);
            }

            template<typename GeneType>
            __device__
            GeneType operator()(
                std::size_t generation,
                GeneType allele,
                curandState& rng_state
            ) const {
                float sigma = current_sigma(generation);
                float noise = sigma * curand_normal(&rng_state);

                GeneType mutated = allele + static_cast<GeneType>(noise);

                if (mutated < static_cast<GeneType>(0)) {
                    mutated = static_cast<GeneType>(0);
                }

                return mutated;
            }
    };

}