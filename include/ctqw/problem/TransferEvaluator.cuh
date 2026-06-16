#pragma once

#include "ctqw/problem/TransferProblem.hpp"
#include "ctqw/sparse/DeviceCsrMatrix.cuh"

namespace ctqw::problem {

    class TransferEvaluator {
        public:
            explicit TransferEvaluator(
                const ctqw::sparse::DeviceCsrMatrix& laplacian
            );

            TransferResult evaluate(const TransferProblem& problem) const;

        private:
            const ctqw::sparse::DeviceCsrMatrix& laplacian;
    };

}