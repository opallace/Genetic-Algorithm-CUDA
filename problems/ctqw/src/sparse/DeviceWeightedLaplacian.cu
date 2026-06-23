#include "sparse/DeviceWeightedLaplacian.cuh"

#include <stdexcept>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "utils/CudaCheck.cuh"

namespace ctqw::sparse {

    namespace {

        void validate_graph_for_laplacian(const ctqw::graph::Graph& graph) {
            if (graph.vertex_count() <= 0) {
                throw std::invalid_argument(
                    "DeviceWeightedLaplacian: graph must have at least one vertex."
                );
            }

            for (const auto& edge : graph.get_edges()) {
                if (edge.u < 0 || edge.u >= graph.vertex_count() || edge.v < 0 || edge.v >= graph.vertex_count()) {
                    throw std::out_of_range(
                        "DeviceWeightedLaplacian: edge vertex index out of range."
                    );
                }

                if (edge.u == edge.v) {
                    throw std::invalid_argument(
                        "DeviceWeightedLaplacian: self-loops are not supported."
                    );
                }
            }
        }

    }

    DeviceWeightedLaplacian::DeviceWeightedLaplacian(
        const ctqw::graph::Graph& graph
    ) {
        upload(graph);
    }

    DeviceWeightedLaplacian::~DeviceWeightedLaplacian() {
        release();
    }

    DeviceWeightedLaplacian::DeviceWeightedLaplacian(
        DeviceWeightedLaplacian&& other
    ) noexcept {
        n_vertices    = other.n_vertices;
        n_edges       = other.n_edges;
        adjacency_nnz = other.adjacency_nnz;

        d_row_offsets = other.d_row_offsets;
        d_neighbors   = other.d_neighbors;
        d_edge_ids    = other.d_edge_ids;

        other.n_vertices    = 0;
        other.n_edges       = 0;
        other.adjacency_nnz = 0;

        other.d_row_offsets = nullptr;
        other.d_neighbors   = nullptr;
        other.d_edge_ids    = nullptr;
    }

    DeviceWeightedLaplacian& DeviceWeightedLaplacian::operator=(
        DeviceWeightedLaplacian&& other
    ) noexcept {
        if (this != &other) {
            release();

            n_vertices    = other.n_vertices;
            n_edges       = other.n_edges;
            adjacency_nnz = other.adjacency_nnz;

            d_row_offsets = other.d_row_offsets;
            d_neighbors   = other.d_neighbors;
            d_edge_ids    = other.d_edge_ids;

            other.n_vertices    = 0;
            other.n_edges       = 0;
            other.adjacency_nnz = 0;

            other.d_row_offsets = nullptr;
            other.d_neighbors   = nullptr;
            other.d_edge_ids    = nullptr;
        }

        return *this;
    }

    void DeviceWeightedLaplacian::upload(
        const ctqw::graph::Graph& graph
    ) {
        release();
        validate_graph_for_laplacian(graph);

        n_vertices    = graph.vertex_count();
        n_edges       = graph.edge_count();
        adjacency_nnz = static_cast<int>(2 * n_edges);

        std::vector<int> degree(static_cast<std::size_t>(n_vertices), 0);

        for (const auto& edge : graph.get_edges()) {
            degree[static_cast<std::size_t>(edge.u)]++;
            degree[static_cast<std::size_t>(edge.v)]++;
        }

        std::vector<int> row_offsets(static_cast<std::size_t>(n_vertices + 1), 0);

        for (int vertex = 0; vertex < n_vertices; vertex++) {
            row_offsets[static_cast<std::size_t>(vertex + 1)] = row_offsets[static_cast<std::size_t>(vertex)] + degree[static_cast<std::size_t>(vertex)];
        }

        std::vector<int> neighbors(static_cast<std::size_t>(adjacency_nnz));
        std::vector<int> edge_ids(static_cast<std::size_t>(adjacency_nnz));
        std::vector<int> cursor = row_offsets;

        for (std::size_t edge_id = 0; edge_id < graph.edge_count(); edge_id++) {
            const auto& edge = graph.get_edges()[edge_id];

            int position_u = cursor[static_cast<std::size_t>(edge.u)]++;
            neighbors[static_cast<std::size_t>(position_u)] = edge.v;
            edge_ids[static_cast<std::size_t>(position_u)] = static_cast<int>(edge_id);

            int position_v = cursor[static_cast<std::size_t>(edge.v)]++;
            neighbors[static_cast<std::size_t>(position_v)] = edge.u;
            edge_ids[static_cast<std::size_t>(position_v)] = static_cast<int>(edge_id);
        }

        CTQW_CUDA_CHECK(cudaMalloc(
            &d_row_offsets,
            sizeof(int) * static_cast<std::size_t>(n_vertices + 1)
        ));

        CTQW_CUDA_CHECK(cudaMemcpy(
            d_row_offsets,
            row_offsets.data(),
            sizeof(int) * static_cast<std::size_t>(n_vertices + 1),
            cudaMemcpyHostToDevice
        ));

        if (adjacency_nnz > 0) {
            CTQW_CUDA_CHECK(cudaMalloc(
                &d_neighbors,
                sizeof(int) * static_cast<std::size_t>(adjacency_nnz)
            ));

            CTQW_CUDA_CHECK(cudaMalloc(
                &d_edge_ids,
                sizeof(int) * static_cast<std::size_t>(adjacency_nnz)
            ));

            CTQW_CUDA_CHECK(cudaMemcpy(
                d_neighbors,
                neighbors.data(),
                sizeof(int) * static_cast<std::size_t>(adjacency_nnz),
                cudaMemcpyHostToDevice
            ));

            CTQW_CUDA_CHECK(cudaMemcpy(
                d_edge_ids,
                edge_ids.data(),
                sizeof(int) * static_cast<std::size_t>(adjacency_nnz),
                cudaMemcpyHostToDevice
            ));
        }
    }

    void DeviceWeightedLaplacian::release() {
        if (d_edge_ids) {
            cudaFree(d_edge_ids);
            d_edge_ids = nullptr;
        }

        if (d_neighbors) {
            cudaFree(d_neighbors);
            d_neighbors = nullptr;
        }

        if (d_row_offsets) {
            cudaFree(d_row_offsets);
            d_row_offsets = nullptr;
        }

        n_vertices    = 0;
        n_edges       = 0;
        adjacency_nnz = 0;
    }

}
