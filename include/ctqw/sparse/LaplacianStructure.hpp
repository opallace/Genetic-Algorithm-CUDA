#pragma once

#include <vector>

#include <cuComplex.h>

#include "ctqw/graph/Graph.hpp"
#include "ctqw/sparse/CsrMatrix.hpp"
#include "ctqw/sparse/LaplacianWeightUpdater.cuh"

namespace ctqw::sparse {

    class LaplacianStructure {
        public:
            static std::vector<EdgeCsrMap> build_edge_map(
                const ctqw::graph::Graph& graph,
                const CsrMatrix<cuComplex>& laplacian
            );

        private:
            static int find_position_in_row(
                const CsrMatrix<cuComplex>& matrix,
                int row,
                int col
            );
    };

}