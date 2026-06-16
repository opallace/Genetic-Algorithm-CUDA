#pragma once

#include <cuComplex.h>
#include <cuda_runtime.h>

#include "ctqw/sparse/CsrMatrix.hpp"

namespace ctqw::sparse {

    class DeviceCsrMatrix {
    public:
        DeviceCsrMatrix() = default;

        explicit DeviceCsrMatrix(const CsrMatrix<cuComplex>& host_matrix) {
            upload(host_matrix);
        }

        ~DeviceCsrMatrix() {
            release();
        }

        DeviceCsrMatrix(const DeviceCsrMatrix&) = delete;
        DeviceCsrMatrix& operator=(const DeviceCsrMatrix&) = delete;

        DeviceCsrMatrix(DeviceCsrMatrix&& other) noexcept;
        DeviceCsrMatrix& operator=(DeviceCsrMatrix&& other) noexcept;

        void upload(const CsrMatrix<cuComplex>& host_matrix);
        void release();

        int rows() const {
            return n_rows;
        }

        int cols() const {
            return n_cols;
        }

        int nonzeros() const {
            return nnz;
        }

        int* row_offsets() const {
            return d_row_offsets;
        }

        int* col_indices() const {
            return d_col_indices;
        }

        cuComplex* values() const {
            return d_values;
        }

    private:
        int n_rows = 0;
        int n_cols = 0;
        int nnz = 0;

        int* d_row_offsets = nullptr;
        int* d_col_indices = nullptr;
        cuComplex* d_values = nullptr;
    };

}