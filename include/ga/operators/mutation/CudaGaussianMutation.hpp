#pragma once

#include <curand_kernel.h>

#include "ga/utils/CudaRandom.cuh"

namespace ga::mutation {

    /**
     * @brief Bounded truncated Gaussian mutation operator for CUDA genetic algorithms.
     *
     * This mutation operator mutates an allele with probability
     * @ref mutation_rate. When mutation occurs, the new allele is sampled from
     * a Gaussian distribution centered at the current allele value, with
     * standard deviation @ref sigma, truncated to the interval
     * [@ref min_value, @ref max_value].
     *
     * The mutation rule, when mutation occurs, is:
     *
     * @code
     * mutated ~ N(allele, sigma^2), with min_value <= mutated <= max_value
     * @endcode
     *
     * If mutation does not occur, the original allele is returned unchanged.
     *
     * This differs from applying Gaussian noise and then clamping the result.
     * A simple clamp creates artificial probability mass at the boundaries,
     * while truncated sampling draws directly from the valid region of the
     * Gaussian distribution.
     */
    class CudaGaussianMutation {
        public:
            /**
             * @brief Minimum allowed allele value after mutation.
             */
            float min_value;

            /**
             * @brief Maximum allowed allele value after mutation.
             */
            float max_value;

            /**
             * @brief Standard deviation of the Gaussian mutation.
             */
            float sigma;

            /**
             * @brief Probability of mutating each allele.
             *
             * A value of 0 disables mutation.
             * A value of 1 mutates every allele.
             */
            float mutation_rate;

            /**
             * @brief Constructs a bounded truncated Gaussian mutation operator.
             *
             * @param min_value_ Minimum allowed allele value.
             * @param max_value_ Maximum allowed allele value.
             * @param sigma_ Standard deviation of the Gaussian mutation.
             * @param mutation_rate_ Probability of mutating each allele.
             */
            __host__ __device__
            CudaGaussianMutation(
                float min_value_,
                float max_value_,
                float sigma_,
                float mutation_rate_
            )
                : min_value(min_value_),
                  max_value(max_value_),
                  sigma(sigma_),
                  mutation_rate(mutation_rate_)
            {}

            /**
             * @brief Applies bounded truncated Gaussian mutation to one allele.
             *
             * With probability @ref mutation_rate, samples a new allele from a
             * Gaussian distribution centered at @p allele and truncated to
             * [@ref min_value, @ref max_value]. Otherwise, returns @p allele
             * unchanged.
             *
             * @param allele Allele value to be possibly mutated.
             * @param rng_state CURAND random state used to sample the mutation.
             *
             * @return Original or mutated allele value.
             */
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
                    const float mutation_draw = curand_uniform(&rng_state);

                    if (mutation_draw > mutation_rate) {
                        return allele;
                    }
                }

                float safe_sigma = sigma;

                if (safe_sigma < eps) {
                    safe_sigma = eps;
                }

                float center = allele;

                if (center < min_value) {
                    center = min_value;
                }

                if (center > max_value) {
                    center = max_value;
                }

                const float alpha = (min_value - center) / safe_sigma;
                const float beta  = (max_value - center) / safe_sigma;

                const float phi_alpha = ga::utils::standard_normal_cdf(alpha);
                const float phi_beta  = ga::utils::standard_normal_cdf(beta);

                const float interval = phi_beta - phi_alpha;

                if (interval <= eps) {
                    return center;
                }

                const float u = curand_uniform(&rng_state);
                const float p = phi_alpha + interval * u;
                const float z = ga::utils::standard_normal_inverse_cdf(p);

                float mutated = center + safe_sigma * z;

                if (mutated < min_value) {
                    mutated = min_value;
                }

                if (mutated > max_value) {
                    mutated = max_value;
                }

                return mutated;
            }
    };

}