#pragma once

#include <cuComplex.h>

namespace ctqw::solver {

    void launch_compute_ctqw_derivative(
        const cuComplex* d_lpsi,
        cuComplex* d_k,
        int n
    );

    void launch_compute_ctqw_derivative_csr(
        const int* d_row_offsets,
        const int* d_col_indices,
        const cuComplex* d_values,
        const cuComplex* d_psi,
        cuComplex* d_k,
        int n
    );

    void launch_make_rk4_temp(
        const cuComplex* d_psi,
        const cuComplex* d_k,
        cuComplex* d_temp,
        int n,
        float scale
    );

    void launch_rk4_update(
        cuComplex* d_psi,
        const cuComplex* d_k1,
        const cuComplex* d_k2,
        const cuComplex* d_k3,
        const cuComplex* d_k4,
        int n,
        float dt
    );

}