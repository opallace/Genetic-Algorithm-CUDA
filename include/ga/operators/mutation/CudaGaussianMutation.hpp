#pragma once

#include <curand_kernel.h>

namespace ga::cuda::mutation {

    class CudaGaussianMutation {
        public:
            float sigma;

            __host__ __device__
            explicit CudaGaussianMutation(float sigma)
                : sigma(sigma) {}

            template<typename GeneType>
            __device__
            GeneType operator()(GeneType allele, curandState& rng_state) const {
                float noise = sigma * curand_normal(&rng_state);
                return allele + static_cast<GeneType>(noise);

            }
            
    };

}