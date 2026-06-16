#include "ctqw/sparse/LaplacianBuilder.hpp"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

namespace ctqw::sparse {

    CsrMatrix<float> LaplacianBuilder::build_real_csr(
        const ctqw::graph::Graph& graph
    ) {
        const int n = graph.n;

        std::vector<std::vector<std::pair<int, float>>> rows(n);
        std::vector<float> degree(n, 0.0f);

        for (const auto& edge : graph.edges) {
            const int u   = edge.u;
            const int v   = edge.v;
            const float w = edge.w;

            degree[u] += std::abs(w);
            degree[v] += std::abs(w);

            rows[u].push_back({v, -w});
            rows[v].push_back({u, -w});
        }

        for (int i = 0; i < n; i++) {
            rows[i].push_back({i, degree[i]});
        }

        CsrMatrix<float> csr;
        csr.rows = n;
        csr.cols = n;
        csr.row_offsets.resize(n + 1);

        int nnz = 0;

        for (int row = 0; row < n; row++) {
            auto& entries = rows[row];

            std::sort(
                entries.begin(),
                entries.end(),
                [](const auto& a, const auto& b) {
                    return a.first < b.first;
                }
            );

            csr.row_offsets[row] = nnz;

            for (const auto& [col, value] : entries) {
                csr.col_indices.push_back(col);
                csr.values.push_back(value);
                nnz++;
            }
        }

        csr.row_offsets[n] = nnz;
        csr.nnz = nnz;

        return csr;
    }

    CsrMatrix<cuComplex> LaplacianBuilder::build_complex_csr(
        const ctqw::graph::Graph& graph
    ) {
        CsrMatrix<float> real = build_real_csr(graph);

        CsrMatrix<cuComplex> complex;

        complex.rows = real.rows;
        complex.cols = real.cols;
        complex.nnz = real.nnz;
        complex.row_offsets = real.row_offsets;
        complex.col_indices = real.col_indices;
        complex.values.resize(real.values.size());

        for (std::size_t i = 0; i < real.values.size(); i++) {
            complex.values[i] = make_cuFloatComplex(real.values[i], 0.0f);
        }

        return complex;
    }

}