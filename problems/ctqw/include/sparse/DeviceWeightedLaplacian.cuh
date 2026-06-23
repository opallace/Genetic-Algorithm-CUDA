#pragma once

#include <cstddef>

#include "graph/Graph.hpp"

namespace ctqw::sparse {

    /**
     * @brief Device representation of a fixed undirected graph Laplacian operator.
     *
     * This class does not store the Laplacian matrix values. The graph topology is
     * stored once as an adjacency-CSR structure:
     *
     *   row_offsets[u] ... row_offsets[u + 1] -> neighbors of u
     *   edge_ids[p]                          -> edge weight index for that neighbor
     *
     * For a current weight vector w, the operator is applied as
     *
     *   (L psi)_u = sum_{v in N(u)} w_uv (psi_u - psi_v)
     *
     * This is equivalent to the weighted graph Laplacian L = D - A, but avoids
     * materializing diagonal and symmetric off-diagonal CSR values for every
     * chromosome evaluated by the genetic algorithm.
     */
    class DeviceWeightedLaplacian {
        public:
            DeviceWeightedLaplacian() = default;

            explicit DeviceWeightedLaplacian(
                const ctqw::graph::Graph& graph
            );

            ~DeviceWeightedLaplacian();

            DeviceWeightedLaplacian(const DeviceWeightedLaplacian&) = delete;
            DeviceWeightedLaplacian& operator=(const DeviceWeightedLaplacian&) = delete;

            DeviceWeightedLaplacian(DeviceWeightedLaplacian&& other) noexcept;
            DeviceWeightedLaplacian& operator=(DeviceWeightedLaplacian&& other) noexcept;

            void upload(const ctqw::graph::Graph& graph);
            void release();

            int vertex_count() const {
                return n_vertices;
            }

            std::size_t edge_count() const {
                return n_edges;
            }

            int adjacency_nonzeros() const {
                return adjacency_nnz;
            }

            const int* row_offsets() const {
                return d_row_offsets;
            }

            const int* neighbors() const {
                return d_neighbors;
            }

            const int* edge_ids() const {
                return d_edge_ids;
            }

        private:
            int n_vertices      = 0;
            std::size_t n_edges = 0;
            int adjacency_nnz   = 0;

            int* d_row_offsets = nullptr;
            int* d_neighbors   = nullptr;
            int* d_edge_ids    = nullptr;
    };

}
