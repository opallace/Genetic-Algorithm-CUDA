#include "ctqw/problem/TransferEvaluator.cuh"

#include <cmath>
#include <vector>

#include <cuda_runtime.h>
#include <cuComplex.h>

#include "ctqw/solver/CtqwRk4Solver.cuh"
#include "ctqw/observable/Probability.hpp"
#include "ctqw/utils/CudaCheck.cuh"

namespace ctqw::problem {

    TransferEvaluator::TransferEvaluator(
        const ctqw::sparse::DeviceCsrMatrix& laplacian
    )
        : laplacian(laplacian)
    {}

    TransferResult TransferEvaluator::evaluate(
        const TransferProblem& problem
        
    ) const {
        const int n = laplacian.rows();

        std::vector<cuComplex> psi(
            static_cast<std::size_t>(n),
            make_cuFloatComplex(0.0f, 0.0f)
        );

        psi[problem.source] = make_cuFloatComplex(1.0f, 0.0f);

        cuComplex* d_psi = nullptr;

        CTQW_CUDA_CHECK(cudaMalloc(
            &d_psi,
            sizeof(cuComplex) * static_cast<std::size_t>(n)
        ));

        CTQW_CUDA_CHECK(cudaMemcpy(
            d_psi,
            psi.data(),
            sizeof(cuComplex) * static_cast<std::size_t>(n),
            cudaMemcpyHostToDevice
        ));

        ctqw::solver::CtqwRk4Solver solver(laplacian);

        solver.evolve(
            d_psi,
            problem.final_time,
            problem.steps
        );

        CTQW_CUDA_CHECK(cudaMemcpy(
            psi.data(),
            d_psi,
            sizeof(cuComplex) * static_cast<std::size_t>(n),
            cudaMemcpyDeviceToHost
        ));

        cudaFree(d_psi);

        TransferResult result;

        result.target_probability = ctqw::observable::probability_at(psi, problem.target);
        result.total_probability  = ctqw::observable::total_probability(psi);

        result.norm_error = std::abs(1.0f - result.total_probability);

        return result;
    }

}