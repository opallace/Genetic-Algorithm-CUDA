#include "ctqw/solver/CtqwKernels.cuh"

#include <cuda_runtime.h>

#include "ctqw/utils/CudaCheck.cuh"

namespace ctqw::solver {

    namespace {

        __global__
        void compute_ctqw_derivative_kernel(
            const cuComplex* lpsi,
            cuComplex* k,
            int n
        ) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;

            if (i >= n) {
                return;
            }

            float a = cuCrealf(lpsi[i]);
            float b = cuCimagf(lpsi[i]);

            k[i] = make_cuFloatComplex(b, -a);
        }

        __global__
        void compute_ctqw_derivative_csr_kernel(
            const int* row_offsets,
            const int* col_indices,
            const cuComplex* values,
            const cuComplex* psi,
            cuComplex* k,
            int n
        ) {
            int row = blockIdx.x * blockDim.x + threadIdx.x;

            if (row >= n) {
                return;
            }

            cuComplex sum = make_cuFloatComplex(0.0f, 0.0f);

            int begin = row_offsets[row];
            int end   = row_offsets[row + 1];

            for (int p = begin; p < end; p++) {
                int col = col_indices[p];

                cuComplex a = values[p];
                cuComplex x = psi[col];

                sum = cuCaddf(sum, cuCmulf(a, x));
            }

            float re = cuCrealf(sum);
            float im = cuCimagf(sum);

            // k = -i * (L psi)
            // se Lpsi = re + i im,
            // então -i Lpsi = im - i re
            k[row] = make_cuFloatComplex(im, -re);
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

    void launch_compute_ctqw_derivative(
        const cuComplex* d_lpsi,
        cuComplex* d_k,
        int n
    ) {
        int threads = 1024;
        int blocks = (n + threads - 1) / threads;

        compute_ctqw_derivative_kernel<<<blocks, threads>>>(
            d_lpsi,
            d_k,
            n
        );

        CTQW_CUDA_CHECK(cudaGetLastError());
    }

    void launch_compute_ctqw_derivative_csr(
        const int* d_row_offsets,
        const int* d_col_indices,
        const cuComplex* d_values,
        const cuComplex* d_psi,
        cuComplex* d_k,
        int n
    ) {
        constexpr int threads = 1024;
        int blocks = (n + threads - 1) / threads;

        compute_ctqw_derivative_csr_kernel<<<blocks, threads>>>(
            d_row_offsets,
            d_col_indices,
            d_values,
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
        int threads = 1024;
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
        int threads = 1024;
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