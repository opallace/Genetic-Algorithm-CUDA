#include "solver/CtqwBatchKernels.cuh"

#include <cmath>
#include <cuda_runtime.h>

#include "utils/CudaCheck.cuh"

namespace ctqw::solver {

    namespace {

        __global__
        void reset_initial_states_batch_kernel(
            cuComplex* psi,
            std::size_t population_size,
            int n,
            int source
        ) {
            const std::size_t id    = blockIdx.x * blockDim.x + threadIdx.x;
            const std::size_t total = population_size * static_cast<std::size_t>(n);

            if (id >= total) {
                return;
            }

            const int u = static_cast<int>(id % n);

            psi[id] = (u == source)
                ? make_cuFloatComplex(1.0f, 0.0f)
                : make_cuFloatComplex(0.0f, 0.0f);
        }

        __global__
        void compute_ctqw_derivative_weighted_laplacian_batch_kernel(
            const int* row_offsets,
            const int* neighbors,
            const int* edge_ids,
            const float* population,
            const cuComplex* psi,
            cuComplex* k,
            std::size_t population_size,
            int n,
            std::size_t edge_count
        ) {
            const std::size_t id    = blockIdx.x * blockDim.x + threadIdx.x;
            const std::size_t total = population_size * static_cast<std::size_t>(n);

            if (id >= total) {
                return;
            }

            const std::size_t individual = id / n;
            const int u = static_cast<int>(id % n);

            const float* weights = population + individual * edge_count;

            const cuComplex* psi_individual = psi + individual * static_cast<std::size_t>(n);

            cuComplex* k_individual = k + individual * static_cast<std::size_t>(n);

            const cuComplex psi_u = psi_individual[u];

            const float psi_u_re = cuCrealf(psi_u);
            const float psi_u_im = cuCimagf(psi_u);

            float lpsi_re = 0.0f;
            float lpsi_im = 0.0f;

            const int begin = row_offsets[u];
            const int end   = row_offsets[u + 1];

            for (int p = begin; p < end; p++) {
                const int v       = neighbors[p];
                const int edge_id = edge_ids[p];

                const float w =  weights[static_cast<std::size_t>(edge_id)];

                const cuComplex psi_v = psi_individual[v];

                lpsi_re += w * (psi_u_re - cuCrealf(psi_v));
                lpsi_im += w * (psi_u_im - cuCimagf(psi_v));
            }

            k_individual[u] = make_cuFloatComplex(lpsi_im, -lpsi_re);
        }

        __global__
        void make_rk4_temp_batch_kernel(
            const cuComplex* psi,
            const cuComplex* k,
            cuComplex* temp,
            std::size_t total_states,
            float scale
        ) {
            const std::size_t id = blockIdx.x * blockDim.x + threadIdx.x;

            if (id >= total_states) {
                return;
            }

            const cuComplex scaled_k =
                make_cuFloatComplex(
                    scale * cuCrealf(k[id]),
                    scale * cuCimagf(k[id])
                );

            temp[id] = cuCaddf(psi[id], scaled_k);
        }

        __global__
        void rk4_update_batch_kernel(
            cuComplex* psi,
            const cuComplex* k1,
            const cuComplex* k2,
            const cuComplex* k3,
            const cuComplex* k4,
            std::size_t total_states,
            float dt
        ) {
            const std::size_t id =
                blockIdx.x * blockDim.x + threadIdx.x;

            if (id >= total_states) {
                return;
            }

            const float re =
                cuCrealf(k1[id])
                + 2.0f * cuCrealf(k2[id])
                + 2.0f * cuCrealf(k3[id])
                + cuCrealf(k4[id]);

            const float im =
                cuCimagf(k1[id])
                + 2.0f * cuCimagf(k2[id])
                + 2.0f * cuCimagf(k3[id])
                + cuCimagf(k4[id]);

            const cuComplex delta =
                make_cuFloatComplex(
                    (dt / 6.0f) * re,
                    (dt / 6.0f) * im
                );

            psi[id] = cuCaddf(psi[id], delta);
        }

        __global__
        void compute_transfer_fitness_batch_kernel(
            const cuComplex* psi,
            double* fitness_values,
            std::size_t population_size,
            int n,
            int target,
            float norm_penalty
        ) {
            const std::size_t individual =
                blockIdx.x * blockDim.x + threadIdx.x;

            if (individual >= population_size) {
                return;
            }

            const cuComplex* psi_individual =
                psi + individual * static_cast<std::size_t>(n);

            const cuComplex target_amplitude =
                psi_individual[target];

            const float target_probability =
                cuCrealf(target_amplitude) * cuCrealf(target_amplitude)
                + cuCimagf(target_amplitude) * cuCimagf(target_amplitude);

            float total_probability = 0.0f;

            for (int u = 0; u < n; u++) {
                const cuComplex amplitude = psi_individual[u];

                const float re = cuCrealf(amplitude);
                const float im = cuCimagf(amplitude);

                total_probability += re * re + im * im;
            }

            const float norm_error =
                fabsf(1.0f - total_probability);

            fitness_values[individual] =
                static_cast<double>(target_probability)
                - static_cast<double>(norm_penalty)
                * static_cast<double>(norm_error);
        }

    }

    void launch_reset_initial_states_batch(
        cuComplex* d_psi,
        std::size_t population_size,
        int n,
        int source
    ) {
        constexpr int threads = 1024;

        const std::size_t total = population_size * static_cast<std::size_t>(n);

        const int blocks = static_cast<int>((total + threads - 1) / threads);

        reset_initial_states_batch_kernel<<<blocks, threads>>>(
            d_psi,
            population_size,
            n,
            source
        );

        CTQW_CUDA_CHECK(cudaGetLastError());
    }

    void launch_compute_ctqw_derivative_weighted_laplacian_batch(
        const int* d_row_offsets,
        const int* d_neighbors,
        const int* d_edge_ids,
        const float* d_population,
        const cuComplex* d_psi,
        cuComplex* d_k,
        std::size_t population_size,
        int n,
        std::size_t edge_count
    ) {
        constexpr int threads = 1024;

        const std::size_t total =
            population_size * static_cast<std::size_t>(n);

        const int blocks =
            static_cast<int>((total + threads - 1) / threads);

        compute_ctqw_derivative_weighted_laplacian_batch_kernel<<<blocks, threads>>>(
            d_row_offsets,
            d_neighbors,
            d_edge_ids,
            d_population,
            d_psi,
            d_k,
            population_size,
            n,
            edge_count
        );

        CTQW_CUDA_CHECK(cudaGetLastError());
    }

    void launch_make_rk4_temp_batch(
        const cuComplex* d_psi,
        const cuComplex* d_k,
        cuComplex* d_temp,
        std::size_t population_size,
        int n,
        float scale
    ) {
        constexpr int threads = 1024;

        const std::size_t total =
            population_size * static_cast<std::size_t>(n);

        const int blocks =
            static_cast<int>((total + threads - 1) / threads);

        make_rk4_temp_batch_kernel<<<blocks, threads>>>(
            d_psi,
            d_k,
            d_temp,
            total,
            scale
        );

        CTQW_CUDA_CHECK(cudaGetLastError());
    }

    void launch_rk4_update_batch(
        cuComplex* d_psi,
        const cuComplex* d_k1,
        const cuComplex* d_k2,
        const cuComplex* d_k3,
        const cuComplex* d_k4,
        std::size_t population_size,
        int n,
        float dt
    ) {
        constexpr int threads = 1024;

        const std::size_t total =
            population_size * static_cast<std::size_t>(n);

        const int blocks =
            static_cast<int>((total + threads - 1) / threads);

        rk4_update_batch_kernel<<<blocks, threads>>>(
            d_psi,
            d_k1,
            d_k2,
            d_k3,
            d_k4,
            total,
            dt
        );

        CTQW_CUDA_CHECK(cudaGetLastError());
    }

    void launch_compute_transfer_fitness_batch(
        const cuComplex* d_psi,
        double* d_fitness_values,
        std::size_t population_size,
        int n,
        int target,
        float norm_penalty
    ) {
        constexpr int threads = 1024;

        const int blocks =
            static_cast<int>((population_size + threads - 1) / threads);

        compute_transfer_fitness_batch_kernel<<<blocks, threads>>>(
            d_psi,
            d_fitness_values,
            population_size,
            n,
            target,
            norm_penalty
        );

        CTQW_CUDA_CHECK(cudaGetLastError());
    }

}