#include "ctqw/solver/CtqwRk4Solver.cuh"

#include <cuda_runtime.h>

#include "ctqw/solver/CtqwKernels.cuh"
#include "ctqw/utils/CudaCheck.cuh"

namespace ctqw::solver {

    CtqwRk4Solver::CtqwRk4Solver(
        const ctqw::sparse::DeviceCsrMatrix& laplacian
    )
        : L(laplacian),
        n(laplacian.rows())
    {
        CTQW_CUSPARSE_CHECK(cusparseCreate(&handle));

        create_sparse_descriptor();
        allocate_workspace();
    }

    CtqwRk4Solver::~CtqwRk4Solver() {
        destroy_dense_descriptors();
        release_workspace();
        destroy_sparse_descriptor();

        if (handle) {
            cusparseDestroy(handle);
            handle = nullptr;
        }
    }

    void CtqwRk4Solver::create_sparse_descriptor() {
        CTQW_CUSPARSE_CHECK(cusparseCreateCsr(
            &matL,
            L.rows(),
            L.cols(),
            L.nonzeros(),
            L.row_offsets(),
            L.col_indices(),
            L.values(),
            CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO,
            CUDA_C_32F
        ));
    }

    void CtqwRk4Solver::destroy_sparse_descriptor() {
        if (matL) {
            cusparseDestroySpMat(matL);
            matL = nullptr;
        }
    }

    void CtqwRk4Solver::allocate_workspace() {
        CTQW_CUDA_CHECK(cudaMalloc(
            &d_lpsi,
            sizeof(cuComplex) * static_cast<std::size_t>(n)
        ));

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
        if (d_buffer) {
            cudaFree(d_buffer);
            d_buffer = nullptr;
            buffer_size = 0;
        }

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

        if (d_lpsi) {
            cudaFree(d_lpsi);
            d_lpsi = nullptr;
        }
    }


    void CtqwRk4Solver::create_dense_descriptors(cuComplex* d_psi) {
        if (vecPsi && descriptor_psi_pointer == d_psi) {
            return;
        }

        destroy_dense_descriptors();

        descriptor_psi_pointer = d_psi;

        CTQW_CUSPARSE_CHECK(cusparseCreateDnVec(
            &vecPsi,
            n,
            d_psi,
            CUDA_C_32F
        ));

        CTQW_CUSPARSE_CHECK(cusparseCreateDnVec(
            &vecTemp,
            n,
            d_temp,
            CUDA_C_32F
        ));

        CTQW_CUSPARSE_CHECK(cusparseCreateDnVec(
            &vecLPsi,
            n,
            d_lpsi,
            CUDA_C_32F
        ));

        if (!d_buffer) {
            cuComplex alpha = make_cuFloatComplex(1.0f, 0.0f);
            cuComplex beta = make_cuFloatComplex(0.0f, 0.0f);

            CTQW_CUSPARSE_CHECK(cusparseSpMV_bufferSize(
                handle,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha,
                matL,
                vecPsi,
                &beta,
                vecLPsi,
                CUDA_C_32F,
                CUSPARSE_SPMV_ALG_DEFAULT,
                &buffer_size
            ));

            CTQW_CUDA_CHECK(cudaMalloc(&d_buffer, buffer_size));
        }
    }

    void CtqwRk4Solver::destroy_dense_descriptors() {
        if (vecPsi) {
            cusparseDestroyDnVec(vecPsi);
            vecPsi = nullptr;
        }

        if (vecTemp) {
            cusparseDestroyDnVec(vecTemp);
            vecTemp = nullptr;
        }

        if (vecLPsi) {
            cusparseDestroyDnVec(vecLPsi);
            vecLPsi = nullptr;
        }

        descriptor_psi_pointer = nullptr;
    }

    void CtqwRk4Solver::compute_derivative(
        const cuComplex* d_input,
        cuComplex* d_k
    ) {
        launch_compute_ctqw_derivative_csr(
            L.row_offsets(),
            L.col_indices(),
            L.values(),
            d_input,
            d_k,
            n
        );
    }

    void CtqwRk4Solver::evolve(
        cuComplex* d_psi,
        float final_time,
        int steps
    ) {
        const float dt = final_time / static_cast<float>(steps);

        for (int step = 0; step < steps; step++) {

            compute_derivative(d_psi, d_k1);

            launch_make_rk4_temp(
                d_psi,
                d_k1,
                d_temp,
                n,
                0.5f * dt
            );

            compute_derivative(d_temp, d_k2);

            launch_make_rk4_temp(
                d_psi,
                d_k2,
                d_temp,
                n,
                0.5f * dt
            );

            compute_derivative(d_temp, d_k3);

            launch_make_rk4_temp(
                d_psi,
                d_k3,
                d_temp,
                n,
                dt
            );

            compute_derivative(d_temp, d_k4);

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