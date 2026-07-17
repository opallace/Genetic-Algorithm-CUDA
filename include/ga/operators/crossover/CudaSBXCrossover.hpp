#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace ga::crossover
{

    /*
     * Simulated Binary Crossover (SBX) for real-coded genetic algorithms.
     *
     * This operator receives two parent alleles and generates two child alleles.
     * The crossover decision itself is not made here; it is controlled outside
     * by pair_crosses[pair_id].
     *
     * eta controls the spread of the generated children:
     *
     *     larger eta -> children closer to the parents
     *     smaller eta -> more exploratory children
     */
    class CudaSBXCrossover
    {
        public:
            float min_value;
            float max_value;
            float eta;

            __host__ __device__
            CudaSBXCrossover(
                float min_value_,
                float max_value_,
                float eta_
            )
                : min_value(min_value_),
                  max_value(max_value_),
                  eta(eta_)
            {}

            __device__
            void operator()(
                float allele_a,
                float allele_b,
                float& child_a,
                float& child_b,
                curandState& rng_state
            ) const
            {
                constexpr float eps = 1.0e-7f;

                float parent_1 = allele_a;
                float parent_2 = allele_b;

                if (fabsf(parent_1 - parent_2) <= eps)
                {
                    child_a = parent_1;
                    child_b = parent_2;
                    return;
                }

                float y1 = fminf(parent_1, parent_2);
                float y2 = fmaxf(parent_1, parent_2);

                float lower = min_value;
                float upper = max_value;

                float rand = curand_uniform(&rng_state);

                rand = fminf(fmaxf(rand, eps), 1.0f - eps);

                float beta  =  1.0f + 2.0f * (y1 - lower) / (y2 - y1);
                float alpha = 2.0f - powf(beta, -(eta + 1.0f));

                float beta_q;

                if (rand <= 1.0f / alpha)
                {
                    beta_q = powf(
                            rand * alpha,
                            1.0f / (eta + 1.0f)
                        );
                }
                else
                {
                    beta_q =
                        powf(
                            1.0f / (2.0f - rand * alpha),
                            1.0f / (eta + 1.0f)
                        );
                }

                float c1 = 0.5f * ((y1 + y2) - beta_q * (y2 - y1));

                beta  = 1.0f + 2.0f * (upper - y2) / (y2 - y1);
                alpha = 2.0f - powf(beta, -(eta + 1.0f));

                if (rand <= 1.0f / alpha) {
                    beta_q =
                        powf(
                            rand * alpha,
                            1.0f / (eta + 1.0f)
                        );

                } else {
                    beta_q =
                        powf(
                            1.0f / (2.0f - rand * alpha),
                            1.0f / (eta + 1.0f)
                        );

                }

                float c2 = 0.5f * ((y1 + y2) + beta_q * (y2 - y1));

                c1 = fminf(fmaxf(c1, lower), upper);
                c2 = fminf(fmaxf(c2, lower),upper);

                /*
                 * Random swap avoids assigning the lower child always to child_a
                 * and the upper child always to child_b.
                 */
                if ((curand(&rng_state) & 1u) == 0u)
                {
                    child_a = c2;
                    child_b = c1;
                }
                else
                {
                    child_a = c1;
                    child_b = c2;
                }
            }
    };

}