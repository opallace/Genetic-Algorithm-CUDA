#include "solver/CtqwKernels.cuh"

#include <cuda_runtime.h>

#include "utils/CudaCheck.cuh"

namespace ctqw::solver {

    namespace {

        __global__
        void compute_ctqw_derivative_weighted_laplacian_kernel(
            const int* row_offsets,
            const int* neighbors,
            const int* edge_ids,
            const float* weights,
            const cuComplex* psi,
            cuComplex* k,
            int n
        ) {
            int u = blockIdx.x * blockDim.x + threadIdx.x;

            if (u >= n) {
                return;
            }

            const cuComplex psi_u = psi[u];
            const float psi_u_re = cuCrealf(psi_u);
            const float psi_u_im = cuCimagf(psi_u);

            float lpsi_re = 0.0f;
            float lpsi_im = 0.0f;

            const int begin = row_offsets[u];
            const int end   = row_offsets[u + 1];

            for (int p = begin; p < end; p++) {
                const int v       = neighbors[p];
                const int edge_id = edge_ids[p];
                const float w     = weights[edge_id];

                const cuComplex psi_v = psi[v];

                lpsi_re += w * (psi_u_re - cuCrealf(psi_v));
                lpsi_im += w * (psi_u_im - cuCimagf(psi_v));
            }

            // k = d psi / dt = -i L psi.
            // If Lpsi = a + ib, then -i Lpsi = b - ia.
            k[u] = make_cuFloatComplex(lpsi_im, -lpsi_re);
        }

        __global__
        void make_rk4_temp_kernel(
            const cuComplex* psi,
            const cuComplex* k,
            cuComplex* temp,
            int n,
            float scale
        ) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;

            if (i >= n) {
                return;
            }

            cuComplex scaled_k = make_cuFloatComplex(
                scale * cuCrealf(k[i]),
                scale * cuCimagf(k[i])
            );

            temp[i] = cuCaddf(psi[i], scaled_k);
        }

        __global__
        void rk4_update_kernel(
            cuComplex* psi,
            const cuComplex* k1,
            const cuComplex* k2,
            const cuComplex* k3,
            const cuComplex* k4,
            int n,
            float dt
        ) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;

            if (i >= n) {
                return;
            }

            float re =
                cuCrealf(k1[i])
                + 2.0f * cuCrealf(k2[i])
                + 2.0f * cuCrealf(k3[i])
                + cuCrealf(k4[i]);

            float im =
                cuCimagf(k1[i])
                + 2.0f * cuCimagf(k2[i])
                + 2.0f * cuCimagf(k3[i])
                + cuCimagf(k4[i]);

            cuComplex delta = make_cuFloatComplex(
                (dt / 6.0f) * re,
                (dt / 6.0f) * im
            );

            psi[i] = cuCaddf(psi[i], delta);
        }

    }

    void launch_compute_ctqw_derivative_weighted_laplacian(
        //CSR
        const int* d_row_offsets,
        const int* d_neighbors,
        const int* d_edge_ids,

        //CROMOSSOME
        const float* d_weights,

        //STATE
        const cuComplex* d_psi,
        cuComplex* d_k,
        int n
    ) {
        constexpr int threads = 1024;
        int blocks = (n + threads - 1) / threads;

        compute_ctqw_derivative_weighted_laplacian_kernel<<<blocks, threads>>>(
            d_row_offsets,
            d_neighbors,
            d_edge_ids,
            d_weights,
            d_psi,
            d_k,
            n
        );

        CTQW_CUDA_CHECK(cudaGetLastError());
    }

    void launch_make_rk4_temp(
        const cuComplex* d_psi,
        const cuComplex* d_k,
        cuComplex* d_temp,
        int n,
        float scale
    ) {
        constexpr int threads = 1024;
        int blocks = (n + threads - 1) / threads;

        make_rk4_temp_kernel<<<blocks, threads>>>(
            d_psi,
            d_k,
            d_temp,
            n,
            scale
        );

        CTQW_CUDA_CHECK(cudaGetLastError());
    }

    void launch_rk4_update(
        cuComplex* d_psi,
        const cuComplex* d_k1,
        const cuComplex* d_k2,
        const cuComplex* d_k3,
        const cuComplex* d_k4,
        int n,
        float dt
    ) {
        constexpr int threads = 1024;
        int blocks = (n + threads - 1) / threads;

        rk4_update_kernel<<<blocks, threads>>>(
            d_psi,
            d_k1,
            d_k2,
            d_k3,
            d_k4,
            n,
            dt
        );

        CTQW_CUDA_CHECK(cudaGetLastError());
    }

}
