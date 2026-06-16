#include "ctqw/sparse/DeviceCsrMatrix.cuh"

#include <utility>

#include "ctqw/utils/CudaCheck.cuh"

namespace ctqw::sparse {

    DeviceCsrMatrix::DeviceCsrMatrix(DeviceCsrMatrix&& other) noexcept {
        n_rows = other.n_rows;
        n_cols = other.n_cols;
        nnz = other.nnz;

        d_row_offsets = other.d_row_offsets;
        d_col_indices = other.d_col_indices;
        d_values = other.d_values;

        other.n_rows = 0;
        other.n_cols = 0;
        other.nnz = 0;

        other.d_row_offsets = nullptr;
        other.d_col_indices = nullptr;
        other.d_values = nullptr;
    }

    DeviceCsrMatrix& DeviceCsrMatrix::operator=(
        DeviceCsrMatrix&& other
    ) noexcept {
        if (this != &other) {
            release();

            n_rows = other.n_rows;
            n_cols = other.n_cols;
            nnz = other.nnz;

            d_row_offsets = other.d_row_offsets;
            d_col_indices = other.d_col_indices;
            d_values = other.d_values;

            other.n_rows = 0;
            other.n_cols = 0;
            other.nnz = 0;

            other.d_row_offsets = nullptr;
            other.d_col_indices = nullptr;
            other.d_values = nullptr;
        }

        return *this;
    }

    void DeviceCsrMatrix::upload(
        const CsrMatrix<cuComplex>& host_matrix
    ) {
        release();

        n_rows = host_matrix.rows;
        n_cols = host_matrix.cols;
        nnz = host_matrix.nnz;

        CTQW_CUDA_CHECK(cudaMalloc(
            &d_row_offsets,
            sizeof(int) * static_cast<std::size_t>(n_rows + 1)
        ));

        CTQW_CUDA_CHECK(cudaMalloc(
            &d_col_indices,
            sizeof(int) * static_cast<std::size_t>(nnz)
        ));

        CTQW_CUDA_CHECK(cudaMalloc(
            &d_values,
            sizeof(cuComplex) * static_cast<std::size_t>(nnz)
        ));

        CTQW_CUDA_CHECK(cudaMemcpy(
            d_row_offsets,
            host_matrix.row_offsets.data(),
            sizeof(int) * static_cast<std::size_t>(n_rows + 1),
            cudaMemcpyHostToDevice
        ));

        CTQW_CUDA_CHECK(cudaMemcpy(
            d_col_indices,
            host_matrix.col_indices.data(),
            sizeof(int) * static_cast<std::size_t>(nnz),
            cudaMemcpyHostToDevice
        ));

        CTQW_CUDA_CHECK(cudaMemcpy(
            d_values,
            host_matrix.values.data(),
            sizeof(cuComplex) * static_cast<std::size_t>(nnz),
            cudaMemcpyHostToDevice
        ));
    }

    void DeviceCsrMatrix::release() {
        if (d_row_offsets) {
            cudaFree(d_row_offsets);
            d_row_offsets = nullptr;
        }

        if (d_col_indices) {
            cudaFree(d_col_indices);
            d_col_indices = nullptr;
        }

        if (d_values) {
            cudaFree(d_values);
            d_values = nullptr;
        }

        n_rows = 0;
        n_cols = 0;
        nnz = 0;
    }

}