#include "problem/StateTransferEvaluator.cuh"

#include <cmath>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>
#include <cuComplex.h>

#include "observable/Probability.hpp"
#include "utils/CudaCheck.cuh"

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

    StateTransferEvaluator::StateTransferEvaluator(
        const ctqw::sparse::DeviceWeightedLaplacian& laplacian
    )
        : laplacian(laplacian),
          n(laplacian.vertex_count()),
          solver(laplacian)
    {
        allocate();
    }

    StateTransferEvaluator::~StateTransferEvaluator() {
        release();
    }

    void StateTransferEvaluator::allocate() {
        h_psi.resize(static_cast<std::size_t>(n));

        CTQW_CUDA_CHECK(cudaMalloc(
            &d_psi,
            sizeof(cuComplex) * static_cast<std::size_t>(n)
        ));
    }

    void StateTransferEvaluator::release() {
        if (d_psi) {
            cudaFree(d_psi);
            d_psi = nullptr;
        }
    }

    void StateTransferEvaluator::reset_initial_state(int source) {
        if (source < 0 || source >= n) {
            throw std::out_of_range(
                "StateTransferEvaluator: source vertex is out of range."
            );
        }

        constexpr int threads = 256;
        int blocks = (n + threads - 1) / threads;

        reset_initial_state_kernel<<<blocks, threads>>>(
            d_psi,
            n,
            source
        );

        CTQW_CUDA_CHECK(cudaGetLastError());
    }

    TransferResult StateTransferEvaluator::evaluate(
        const TransferProblem& problem,
        const float* d_weights
        
    ) {
        if (!d_weights) {
            throw std::invalid_argument(
                "StateTransferEvaluator: d_weights is null."
            );
        }

        if (problem.target < 0 || problem.target >= n) {
            throw std::out_of_range(
                "StateTransferEvaluator: target vertex is out of range."
            );
        }

        reset_initial_state(problem.source);

        solver.evolve(
            d_psi,
            d_weights,
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

        result.target_probability = ctqw::observable::probability_at(h_psi, problem.target);
        result.total_probability  = ctqw::observable::total_probability(h_psi);
        result.norm_error         = std::abs(1.0f - result.total_probability);

        return result;
    }

}
