#pragma once

#include <cstddef>
#include <vector>

#include "ctqw/problem/TransferProblem.hpp"
#include "ctqw/problem/ReusableTransferEvaluator.cuh"
#include "ctqw/sparse/DeviceCsrMatrix.cuh"
#include "ctqw/sparse/LaplacianWeightUpdater.cuh"

namespace ctqw::fitness {

    class CtqwFitnessEvaluator {
        public:
            CtqwFitnessEvaluator(
                float norm_penalty,
                ctqw::problem::TransferProblem problem,
                ctqw::sparse::DeviceCsrMatrix& device_laplacian,
                ctqw::sparse::LaplacianWeightUpdater& laplacian_updater
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

            ctqw::sparse::DeviceCsrMatrix& device_laplacian;
            ctqw::sparse::LaplacianWeightUpdater& laplacian_updater;

            ctqw::problem::ReusableTransferEvaluator transfer_evaluator;

            std::vector<double> h_fitness_values;

            double evaluate_one_individual(
                const float* d_weights,
                std::size_t chromosome_size
            );
    };

}