#include "ctqw/sparse/LaplacianStructure.hpp"

#include <stdexcept>
#include <string>

namespace ctqw::sparse {

    int LaplacianStructure::find_position_in_row(
        const CsrMatrix<cuComplex>& matrix,
        int row,
        int col
    ) {
        if (row < 0 || row >= matrix.rows) {
            throw std::out_of_range(
                "LaplacianStructure: row index out of range."
            );
        }

        const int begin = matrix.row_offsets[row];
        const int end = matrix.row_offsets[row + 1];

        for (int position = begin; position < end; position++) {
            if (matrix.col_indices[position] == col) {
                return position;
            }
        }

        throw std::runtime_error(
            "LaplacianStructure: entry (" +
            std::to_string(row) + ", " +
            std::to_string(col) +
            ") was not found in CSR matrix."
        );
    }

    std::vector<EdgeCsrMap> LaplacianStructure::build_edge_map(
        const ctqw::graph::Graph& graph,
        const CsrMatrix<cuComplex>& laplacian
    ) {
        if (laplacian.rows != graph.n || laplacian.cols != graph.n) {
            throw std::invalid_argument(
                "LaplacianStructure: graph size and Laplacian size do not match."
            );
        }

        std::vector<EdgeCsrMap> edge_map;
        edge_map.reserve(graph.edges.size());

        for (const auto& edge : graph.edges) {
            const int u = edge.u;
            const int v = edge.v;

            if (u < 0 || u >= graph.n || v < 0 || v >= graph.n) {
                throw std::out_of_range(
                    "LaplacianStructure: edge vertex index out of range."
                );
            }

            EdgeCsrMap map;

            map.off_uv = find_position_in_row(
                laplacian,
                u,
                v
            );

            map.off_vu = find_position_in_row(
                laplacian,
                v,
                u
            );

            map.diag_u = find_position_in_row(
                laplacian,
                u,
                u
            );

            map.diag_v = find_position_in_row(
                laplacian,
                v,
                v
            );

            edge_map.push_back(map);
        }

        return edge_map;
    }

}