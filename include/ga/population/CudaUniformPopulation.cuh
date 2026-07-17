#pragma once

#include <curand_kernel.h>

namespace ga::population {

    class CudaUniformPopulation {
        public:
            float min_value;
            float max_value;

            __host__ __device__
            CudaUniformPopulation(
                float min_value_,
                float max_value_
            )
                : min_value(min_value_),
                  max_value(max_value_)
            {}

            template<typename GeneType>
            __device__
            GeneType generate(curandState& rng_state) const {
                const float u = curand_uniform(&rng_state);

                const float value = min_value + (max_value - min_value) * u;

                return static_cast<GeneType>(value);
            }
    };

}