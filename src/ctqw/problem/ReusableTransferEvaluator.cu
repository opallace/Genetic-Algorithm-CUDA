#include "ctqw/problem/ReusableTransferEvaluator.cuh"

#include <cmath>
#include <vector>

#include <cuda_runtime.h>
#include <cuComplex.h>

#include "ctqw/observable/Probability.hpp"
#include "ctqw/utils/CudaCheck.cuh"

namespace ctqw::problem {

    namespace {

        __global__
        void reset_initial_state_kernel(
            cuComplex* psi,
            int n,
            int source
        ) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;

            if (i >= n) {
                return;
            }

            psi[i] = (i == source)
                ? make_cuFloatComplex(1.0f, 0.0f)
                : make_cuFloatComplex(0.0f, 0.0f);
        }

    }

    ReusableTransferEvaluator::ReusableTransferEvaluator(
        const ctqw::sparse::DeviceCsrMatrix& laplacian
    )
        : laplacian(laplacian),
          n(laplacian.rows()),
          solver(laplacian)
    {
        allocate();
    }

    ReusableTransferEvaluator::~ReusableTransferEvaluator() {
        release();
    }

    void ReusableTransferEvaluator::allocate() {
        h_psi.resize(static_cast<std::size_t>(n));

        CTQW_CUDA_CHECK(cudaMalloc(
            &d_psi,
            sizeof(cuComplex) * static_cast<std::size_t>(n)
        ));
    }

    void ReusableTransferEvaluator::release() {
        if (d_psi) {
            cudaFree(d_psi);
            d_psi = nullptr;
        }
    }

    void ReusableTransferEvaluator::reset_initial_state(int source) {
        constexpr int threads = 256;
        int blocks = (n + threads - 1) / threads;

        reset_initial_state_kernel<<<blocks, threads>>>(
            d_psi,
            n,
            source
        );

        CTQW_CUDA_CHECK(cudaGetLastError());
    }

    TransferResult ReusableTransferEvaluator::evaluate(
        const TransferProblem& problem
    ) {
        reset_initial_state(problem.source);

        solver.evolve(
            d_psi,
            problem.final_time,
            problem.steps
        );

        CTQW_CUDA_CHECK(cudaMemcpy(
            h_psi.data(),
            d_psi,
            sizeof(cuComplex) * static_cast<std::size_t>(n),
            cudaMemcpyDeviceToHost
        ));

        TransferResult result;

        result.target_probability =
            ctqw::observable::probability_at(h_psi, problem.target);

        result.total_probability =
            ctqw::observable::total_probability(h_psi);

        result.norm_error =
            std::abs(1.0f - result.total_probability);

        return result;
    }

}