#pragma once

#include <cstddef>
#include <vector>

#include "problem/TransferProblem.hpp"
#include "problem/StateTransferEvaluator.cuh"
#include "sparse/DeviceWeightedLaplacian.cuh"

namespace ctqw::fitness {

    /**
     * @brief Fitness adapter between the CUDA GA and the CTQW transfer problem.
     *
     * Each chromosome is interpreted as the current edge-weight vector for a
     * fixed graph topology. The Laplacian is not materialized as CSR values;
     * the solver applies L(w) directly from the graph adjacency structure.
     */
    class CtqwFitnessEvaluator {
        public:
            CtqwFitnessEvaluator(
                float norm_penalty,
                ctqw::problem::TransferProblem problem,
                const ctqw::sparse::DeviceWeightedLaplacian& laplacian
            );

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
            ctqw::problem::StateTransferEvaluator transfer_evaluator;

            std::vector<double> h_fitness_values;

            double evaluate_one_individual(const float* d_weights);
    };

}
