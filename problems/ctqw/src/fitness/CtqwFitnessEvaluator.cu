#include "fitness/CtqwFitnessEvaluator.cuh"

#include <stdexcept>

#include <cuda_runtime.h>

#include "solver/CtqwBatchKernels.cuh"
#include "utils/CudaCheck.cuh"

namespace ctqw::fitness {

    CtqwFitnessEvaluator::CtqwFitnessEvaluator(
        std::size_t max_population_size,
        float norm_penalty,
        ctqw::problem::TransferProblem problem,
        const ctqw::sparse::DeviceWeightedLaplacian& laplacian
    )
        : norm_penalty(norm_penalty),
          problem(problem),
          laplacian(laplacian),
          n(laplacian.vertex_count())
    {
        reserve_workspace(max_population_size);
    }

    CtqwFitnessEvaluator::~CtqwFitnessEvaluator() {
        release_workspace();
    }

    void CtqwFitnessEvaluator::reserve_workspace(
        std::size_t max_population_size

    ) {
        if (max_population_size == 0) {
            throw std::invalid_argument(
                "CtqwFitnessEvaluator: max_population_size must be greater than zero."
            );
        }

        if (max_population_size <= allocated_population_size) {
            return;
        }

        release_workspace();

        const std::size_t total_states = max_population_size * static_cast<std::size_t>(n);

        CTQW_CUDA_CHECK(cudaMalloc(
            &d_psi,
            sizeof(cuComplex) * total_states
        ));

        CTQW_CUDA_CHECK(cudaMalloc(
            &d_temp,
            sizeof(cuComplex) * total_states
        ));

        CTQW_CUDA_CHECK(cudaMalloc(
            &d_k1,
            sizeof(cuComplex) * total_states
        ));

        CTQW_CUDA_CHECK(cudaMalloc(
            &d_k2,
            sizeof(cuComplex) * total_states
        ));

        CTQW_CUDA_CHECK(cudaMalloc(
            &d_k3,
            sizeof(cuComplex) * total_states
        ));

        CTQW_CUDA_CHECK(cudaMalloc(
            &d_k4,
            sizeof(cuComplex) * total_states
        ));

        allocated_population_size = max_population_size;
    }

    void CtqwFitnessEvaluator::release_workspace() {
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

        if (d_psi) {
            cudaFree(d_psi);
            d_psi = nullptr;
        }

        allocated_population_size = 0;
    }

    void CtqwFitnessEvaluator::evaluate_population(
        const float* d_population,
        double* d_fitness_values,
        std::size_t population_size,
        std::size_t chromosome_size
    ) {
        if (!d_population) {
            throw std::invalid_argument(
                "CtqwFitnessEvaluator: d_population is null."
            );
        }

        if (!d_fitness_values) {
            throw std::invalid_argument(
                "CtqwFitnessEvaluator: d_fitness_values is null."
            );
        }

        if (population_size == 0) {
            throw std::invalid_argument(
                "CtqwFitnessEvaluator: population_size must be greater than zero."
            );
        }

        if (population_size > allocated_population_size) {
            throw std::invalid_argument(
                "CtqwFitnessEvaluator: population_size exceeds reserved workspace size."
            );
        }

        if (chromosome_size == 0) {
            throw std::invalid_argument(
                "CtqwFitnessEvaluator: chromosome_size must be greater than zero."
            );
        }

        if (chromosome_size != laplacian.edge_count()) {
            throw std::invalid_argument(
                "CtqwFitnessEvaluator: chromosome_size must match graph edge count."
            );
        }

        if (problem.source < 0 || problem.source >= n) {
            throw std::out_of_range(
                "CtqwFitnessEvaluator: source vertex is out of range."
            );
        }

        if (problem.target < 0 || problem.target >= n) {
            throw std::out_of_range(
                "CtqwFitnessEvaluator: target vertex is out of range."
            );
        }

        if (problem.steps <= 0) {
            throw std::invalid_argument(
                "CtqwFitnessEvaluator: steps must be positive."
            );
        }

        if (problem.final_time <= 0.0f) {
            throw std::invalid_argument(
                "CtqwFitnessEvaluator: final_time must be positive."
            );
        }

        evolve_population(
            d_population,
            population_size,
            chromosome_size
        );

        ctqw::solver::launch_compute_transfer_fitness_batch(
            d_psi,
            d_fitness_values,
            population_size,
            n,
            problem.target,
            norm_penalty
        );

        CTQW_CUDA_CHECK(cudaDeviceSynchronize());
    }

    void CtqwFitnessEvaluator::evolve_population(
        const float* d_population,
        std::size_t population_size,
        std::size_t chromosome_size
    ) {
        ctqw::solver::launch_reset_initial_states_batch(
            d_psi,
            population_size,
            n,
            problem.source
        );

        const float dt = problem.final_time / static_cast<float>(problem.steps);

        for (int step = 0; step < problem.steps; step++) {

            ctqw::solver::launch_compute_ctqw_derivative_weighted_laplacian_batch(
                laplacian.row_offsets(),
                laplacian.neighbors(),
                laplacian.edge_ids(),
                d_population,
                d_psi,
                d_k1,
                population_size,
                n,
                chromosome_size
            );

            ctqw::solver::launch_make_rk4_temp_batch(
                d_psi,
                d_k1,
                d_temp,
                population_size,
                n,
                0.5f * dt
            );

            ctqw::solver::launch_compute_ctqw_derivative_weighted_laplacian_batch(
                laplacian.row_offsets(),
                laplacian.neighbors(),
                laplacian.edge_ids(),
                d_population,
                d_temp,
                d_k2,
                population_size,
                n,
                chromosome_size
            );

            ctqw::solver::launch_make_rk4_temp_batch(
                d_psi,
                d_k2,
                d_temp,
                population_size,
                n,
                0.5f * dt
            );

            ctqw::solver::launch_compute_ctqw_derivative_weighted_laplacian_batch(
                laplacian.row_offsets(),
                laplacian.neighbors(),
                laplacian.edge_ids(),
                d_population,
                d_temp,
                d_k3,
                population_size,
                n,
                chromosome_size
            );

            ctqw::solver::launch_make_rk4_temp_batch(
                d_psi,
                d_k3,
                d_temp,
                population_size,
                n,
                dt
            );

            ctqw::solver::launch_compute_ctqw_derivative_weighted_laplacian_batch(
                laplacian.row_offsets(),
                laplacian.neighbors(),
                laplacian.edge_ids(),
                d_population,
                d_temp,
                d_k4,
                population_size,
                n,
                chromosome_size
            );

            ctqw::solver::launch_rk4_update_batch(
                d_psi,
                d_k1,
                d_k2,
                d_k3,
                d_k4,
                population_size,
                n,
                dt
            );
        }
    }

}