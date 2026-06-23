#pragma once

#include <curand_kernel.h>

namespace ga::crossover {

    /**
     * @brief Uniform crossover operator for CUDA genetic algorithms.
     *
     * This crossover operator receives two parent alleles and randomly selects
     * one of them to be inherited by the offspring.
     *
     * For each allele position, the child allele has approximately 50% chance
     * of being copied from the first parent and 50% chance of being copied from
     * the second parent.
     *
     * This operator is intended to be called inside CUDA kernels, using a
     * device-side CURAND random state.
     */
    class CudaUniformCrossover {
        public:

            /**
             * @brief Applies uniform crossover to two parent alleles.
             *
             * Randomly chooses either @p allele_a or @p allele_b with equal
             * probability and returns the selected value as the child allele.
             *
             * @tparam GeneType Type of the gene/allele.
             *
             * @param allele_a Allele from the first parent.
             * @param allele_b Allele from the second parent.
             * @param rng_state CURAND random state used to generate the random choice.
             *
             * @return The selected allele, either @p allele_a or @p allele_b.
             *
             * @note This function is marked as `__device__` and must be called
             * from CUDA device code.
             */
            template<typename GeneType>
            __device__
            GeneType operator()(
                GeneType allele_a,
                GeneType allele_b,
                curandState& rng_state
            ) {
                bool choose_a = (curand(&rng_state) % 2) == 0;
                return choose_a ? allele_a : allele_b;
            }

    };

}