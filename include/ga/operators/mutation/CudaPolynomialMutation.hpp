#pragma once

#include <curand_kernel.h>

namespace ga::mutation {

    /**
     * Polynomial mutation operator for real-coded CUDA genetic algorithms.
     *
     * This mutation is commonly paired with SBX. The distribution index eta
     * controls the mutation step size.
     *
     * Larger eta values produce smaller local perturbations.
     * Smaller eta values produce larger exploratory perturbations.
     */
    class CudaPolynomialMutation {
        public:
            float min_value;
            float max_value;
            float eta;
            float mutation_rate;

            __host__ __device__
            CudaPolynomialMutation(
                float min_value_,
                float max_value_,
                float eta_,
                float mutation_rate_
            )
                : min_value(min_value_),
                  max_value(max_value_),
                  eta(eta_),
                  mutation_rate(mutation_rate_)
            {}

            __device__
            float operator()(
                float allele,
                curandState& rng_state
            ) const {
                constexpr float eps = 1.0e-7f;

                if (mutation_rate <= 0.0f) {
                    return allele;
                }

                if (mutation_rate < 1.0f) {
                    float mutation_draw = curand_uniform(&rng_state);

                    if (mutation_draw > mutation_rate) {
                        return allele;
                    }
                }

                float lower = min_value;
                float upper = max_value;

                float range = upper - lower;

                if (range <= eps) {
                    return lower;
                }

                float y = fminf(fmaxf(allele, lower), upper);

                float delta_1 = (y - lower) / range;
                float delta_2 = (upper - y) / range;

                float rand = curand_uniform(&rng_state);

                rand = fminf(rand, 1.0f - eps);

                float mutation_power = 1.0f / (eta + 1.0f);

                float delta_q;

                if (rand <= 0.5f) {
                    float xy = 1.0f - delta_1;

                    float value = 2.0f * rand + (1.0f - 2.0f * rand) * powf(xy, eta + 1.0f);

                    delta_q = powf(value, mutation_power) - 1.0f;

                } else {
                    float xy = 1.0f - delta_2;

                    float value =
                        2.0f * (1.0f - rand) +
                        2.0f * (rand - 0.5f) *
                        powf(xy, eta + 1.0f);

                    delta_q = 1.0f - powf(value, mutation_power);
                }

                y += delta_q * range;

                return fminf(fmaxf(y, lower), upper);
            }
    };

}