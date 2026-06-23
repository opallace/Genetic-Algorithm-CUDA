#include "solver/CtqwRk4Solver.cuh"

#include <stdexcept>
#include <cuda_runtime.h>

#include "solver/CtqwKernels.cuh"
#include "utils/CudaCheck.cuh"

namespace ctqw::solver {

    CtqwRk4Solver::CtqwRk4Solver(
        const ctqw::sparse::DeviceWeightedLaplacian& laplacian
    )
        : L(laplacian),
          n(laplacian.vertex_count())
    {
        allocate_workspace();
    }

    CtqwRk4Solver::~CtqwRk4Solver() {
        release_workspace();
    }

    void CtqwRk4Solver::allocate_workspace() {
        CTQW_CUDA_CHECK(cudaMalloc(
            &d_temp,
            sizeof(cuComplex) * static_cast<std::size_t>(n)
        ));

        CTQW_CUDA_CHECK(cudaMalloc(
            &d_k1,
            sizeof(cuComplex) * static_cast<std::size_t>(n)
        ));

        CTQW_CUDA_CHECK(cudaMalloc(
            &d_k2,
            sizeof(cuComplex) * static_cast<std::size_t>(n)
        ));

        CTQW_CUDA_CHECK(cudaMalloc(
            &d_k3,
            sizeof(cuComplex) * static_cast<std::size_t>(n)
        ));

        CTQW_CUDA_CHECK(cudaMalloc(
            &d_k4,
            sizeof(cuComplex) * static_cast<std::size_t>(n)
        ));
    }

    void CtqwRk4Solver::release_workspace() {
        if (d_k4) {
            cudaFree(d_k4);
            d_k4 = nullptr;
        }

        if (d_k3) {
            cudaFree(d_k3);
            d_k3 = nullptr;
        }

        if (d_k2) {
            cudaFree(d_k2);
            d_k2 = nullptr;
        }

        if (d_k1) {
            cudaFree(d_k1);
            d_k1 = nullptr;
        }

        if (d_temp) {
            cudaFree(d_temp);
            d_temp = nullptr;
        }
    }

    void CtqwRk4Solver::compute_derivative(
        const float* d_weights,
        const cuComplex* d_input,
        cuComplex* d_k
    ) {
        launch_compute_ctqw_derivative_weighted_laplacian(
            L.row_offsets(),
            L.neighbors(),
            L.edge_ids(),
            d_weights,
            d_input,
            d_k,
            n
        );
    }

    void CtqwRk4Solver::evolve(
        cuComplex* d_psi,
        const float* d_weights,
        float final_time,
        int steps
    ) {
        if (!d_psi) {
            throw std::invalid_argument(
                "CtqwRk4Solver::evolve: d_psi is null."
            );
        }

        if (!d_weights) {
            throw std::invalid_argument(
                "CtqwRk4Solver::evolve: d_weights is null."
            );
        }

        if (steps <= 0) {
            throw std::invalid_argument(
                "CtqwRk4Solver::evolve: steps must be positive."
            );
        }

        const float dt = final_time / static_cast<float>(steps);

        for (int step = 0; step < steps; step++) {
            compute_derivative(d_weights, d_psi, d_k1);

            launch_make_rk4_temp(
                d_psi,
                d_k1,
                d_temp,
                n,
                0.5f * dt
            );

            compute_derivative(d_weights, d_temp, d_k2);

            launch_make_rk4_temp(
                d_psi,
                d_k2,
                d_temp,
                n,
                0.5f * dt
            );

            compute_derivative(d_weights, d_temp, d_k3);

            launch_make_rk4_temp(
                d_psi,
                d_k3,
                d_temp,
                n,
                dt
            );

            compute_derivative(d_weights, d_temp, d_k4);

            launch_rk4_update(
                d_psi,
                d_k1,
                d_k2,
                d_k3,
                d_k4,
                n,
                dt
            );
        }
    }

}
