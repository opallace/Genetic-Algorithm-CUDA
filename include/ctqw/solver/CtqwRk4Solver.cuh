#pragma once

#include <cstddef>

#include <cuComplex.h>
#include <cusparse.h>

#include "ctqw/sparse/DeviceCsrMatrix.cuh"

namespace ctqw::solver {

    class CtqwRk4Solver {
    public:
        explicit CtqwRk4Solver(
            const ctqw::sparse::DeviceCsrMatrix& laplacian
        );

        ~CtqwRk4Solver();

        CtqwRk4Solver(const CtqwRk4Solver&) = delete;
        CtqwRk4Solver& operator=(const CtqwRk4Solver&) = delete;

        void evolve(
            cuComplex* d_psi,
            float final_time,
            int steps
        );

    private:
        void allocate_workspace();
        void release_workspace();

        void create_sparse_descriptor();
        void destroy_sparse_descriptor();

        void create_dense_descriptors(cuComplex* d_psi);
        void destroy_dense_descriptors();

        void compute_derivative(
            const cuComplex* d_input,
            cuComplex* d_k
        );

    private:
        const ctqw::sparse::DeviceCsrMatrix& L;

        int n = 0;

        cusparseHandle_t handle = nullptr;
        cusparseSpMatDescr_t matL = nullptr;

        cuComplex* d_lpsi = nullptr;
        cuComplex* d_temp = nullptr;

        cuComplex* d_k1 = nullptr;
        cuComplex* d_k2 = nullptr;
        cuComplex* d_k3 = nullptr;
        cuComplex* d_k4 = nullptr;

        void* d_buffer = nullptr;
        std::size_t buffer_size = 0;

        cuComplex* descriptor_psi_pointer = nullptr;
        cusparseDnVecDescr_t vecPsi = nullptr;
        cusparseDnVecDescr_t vecTemp = nullptr;
        cusparseDnVecDescr_t vecLPsi = nullptr;
    };

}