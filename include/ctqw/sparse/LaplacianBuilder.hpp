#pragma once

#include <cuComplex.h>

#include "ctqw/graph/Graph.hpp"
#include "ctqw/sparse/CsrMatrix.hpp"

namespace ctqw::sparse {

    class LaplacianBuilder {
    public:
        static CsrMatrix<float> build_real_csr(
            const ctqw::graph::Graph& graph
        );

        static CsrMatrix<cuComplex> build_complex_csr(
            const ctqw::graph::Graph& graph
        );
    };

}