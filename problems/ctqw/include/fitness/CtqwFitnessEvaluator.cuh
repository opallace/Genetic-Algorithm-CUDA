#pragma once

#include <cstddef>

#include <cuComplex.h>

#include "problem/TransferProblem.hpp"
#include "sparse/DeviceWeightedLaplacian.cuh"

namespace ctqw::fitness {

    class CtqwFitnessEvaluator {
        public:
            CtqwFitnessEvaluator(
                std::size_t max_population_size,
                float norm_penalty,
                ctqw::problem::TransferProblem problem,
                const ctqw::sparse::DeviceWeightedLaplacian& laplacian
            );

            ~CtqwFitnessEvaluator();

            CtqwFitnessEvaluator(const CtqwFitnessEvaluator&) = delete;
            CtqwFitnessEvaluator& operator=(const CtqwFitnessEvaluator&) = delete;

            CtqwFitnessEvaluator(CtqwFitnessEvaluator&&) = delete;
            CtqwFitnessEvaluator& operator=(CtqwFitnessEvaluator&&) = delete;

            void evaluate_population(
                const float* d_population,
                double* d_fitness_values,
                std::size_t population_size,
                std::size_t chromosome_size
            );

        private:
            float norm_penalty;

            ctqw::problem::TransferProblem problem;
            const ctqw::sparse::DeviceWeightedLaplacian& laplacian;

            int n = 0;
            
            std::size_t allocated_population_size = 0;

            cuComplex* d_psi  = nullptr;
            cuComplex* d_temp = nullptr;
            cuComplex* d_k1   = nullptr;
            cuComplex* d_k2   = nullptr;
            cuComplex* d_k3   = nullptr;
            cuComplex* d_k4   = nullptr;

            void reserve_workspace(std::size_t max_population_size);
            void release_workspace();

            void evolve_population(
                const float* d_population,
                std::size_t population_size,
                std::size_t chromosome_size
            );
    };

}