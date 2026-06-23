#pragma once

#include <cuComplex.h>

namespace ctqw::solver {

    /**
     * @brief Computes k = -i L(w) psi using the fixed graph adjacency structure.
     *
     * The Laplacian is applied without materializing a CSR matrix:
     *
     *   (L psi)_u = sum_{v in N(u)} w_uv (psi_u - psi_v)
     */
    void launch_compute_ctqw_derivative_weighted_laplacian(
        const int* d_row_offsets,
        const int* d_neighbors,
        const int* d_edge_ids,
        const float* d_weights,
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
