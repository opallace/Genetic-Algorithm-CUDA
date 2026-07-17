#pragma once

#include <curand_kernel.h>

#include "ga/utils/CudaRandom.cuh"

namespace ga::population {

    class CudaTruncatedGaussianPopulation {
        public:
            float min_value;
            float max_value;
            float sigma;

            __host__ __device__
            CudaTruncatedGaussianPopulation(
                float min_value_,
                float max_value_,
                float sigma_
            )
                : min_value(min_value_),
                  max_value(max_value_),
                  sigma(sigma_)
            {}

            template<typename GeneType>
            __device__
            GeneType generate(curandState& rng_state) const {
                constexpr float eps = 1.0e-7f;

                const float mean = 0.5f * (min_value + max_value);

                float safe_sigma = sigma;

                if (safe_sigma < eps) {
                    safe_sigma = eps;
                }

                const float alpha = (min_value - mean) / safe_sigma;
                const float beta  = (max_value - mean) / safe_sigma;

                const float phi_alpha = ga::utils::standard_normal_cdf(alpha);
                const float phi_beta  = ga::utils::standard_normal_cdf(beta);

                const float interval = phi_beta - phi_alpha;

                if (interval <= eps) {
                    return mean;
                }

                const float u = curand_uniform(&rng_state);

                const float p = phi_alpha + interval * u;

                const float z = ga::utils::standard_normal_inverse_cdf(p);

                return mean + safe_sigma * z;
            }
    };

}