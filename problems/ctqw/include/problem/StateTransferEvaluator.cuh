#pragma once

#include <vector>

#include <cuComplex.h>

#include "problem/TransferProblem.hpp"
#include "solver/CtqwRk4Solver.cuh"
#include "sparse/DeviceWeightedLaplacian.cuh"

namespace ctqw::problem {

    /**
     * @brief Reusable evaluator for CTQW state transfer on a fixed graph topology.
     *
     * The evaluator owns the device state vector and a reusable RK4 solver. Each
     * evaluation receives the current edge weights directly from a GA chromosome.
     */
    class StateTransferEvaluator {
        public:
            explicit StateTransferEvaluator(
                const ctqw::sparse::DeviceWeightedLaplacian& laplacian
            );

            ~StateTransferEvaluator();

            StateTransferEvaluator(const StateTransferEvaluator&) = delete;
            StateTransferEvaluator& operator=(const StateTransferEvaluator&) = delete;

            TransferResult evaluate(
                const TransferProblem& problem,
                const float* d_weights
            );

        private:
            void allocate();
            void release();
            void reset_initial_state(int source);

        private:
            const ctqw::sparse::DeviceWeightedLaplacian& laplacian;
            int n = 0;

            cuComplex* d_psi = nullptr;
            std::vector<cuComplex> h_psi;

            ctqw::solver::CtqwRk4Solver solver;
    };

}
