#pragma once

#include <vector>

#include <cuComplex.h>

#include "ctqw/problem/TransferProblem.hpp"
#include "ctqw/sparse/DeviceCsrMatrix.cuh"
#include "ctqw/solver/CtqwRk4Solver.cuh"

namespace ctqw::problem {

    class ReusableTransferEvaluator {
        public:
            explicit ReusableTransferEvaluator(
                const ctqw::sparse::DeviceCsrMatrix& laplacian
            );

            ~ReusableTransferEvaluator();

            ReusableTransferEvaluator(const ReusableTransferEvaluator&) = delete;
            ReusableTransferEvaluator& operator=(const ReusableTransferEvaluator&) = delete;

            TransferResult evaluate(const TransferProblem& problem);

        private:
            void allocate();
            void release();
            void reset_initial_state(int source);

        private:
            const ctqw::sparse::DeviceCsrMatrix& laplacian;
            int n = 0;

            cuComplex* d_psi = nullptr;
            std::vector<cuComplex> h_psi;

            ctqw::solver::CtqwRk4Solver solver;
    };

}