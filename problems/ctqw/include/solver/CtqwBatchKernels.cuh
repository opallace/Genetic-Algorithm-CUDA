#pragma once

#include <cstddef>
#include <cuComplex.h>

namespace ctqw::solver {

    void launch_reset_initial_states_batch(
        cuComplex* d_psi,
        std::size_t population_size,
        int n,
        int source
    );

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
    );

    void launch_make_rk4_temp_batch(
        const cuComplex* d_psi,
        const cuComplex* d_k,
        cuComplex* d_temp,
        std::size_t population_size,
        int n,
        float scale
    );

    void launch_rk4_update_batch(
        cuComplex* d_psi,
        const cuComplex* d_k1,
        const cuComplex* d_k2,
        const cuComplex* d_k3,
        const cuComplex* d_k4,
        std::size_t population_size,
        int n,
        float dt
    );

    void launch_compute_transfer_fitness_batch(
        const cuComplex* d_psi,
        double* d_fitness_values,
        std::size_t population_size,
        int n,
        int target,
        float norm_penalty
    );

}