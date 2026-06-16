#pragma once

#include <cstddef>
#include <vector>

#include <cuComplex.h>
#include <cuda_runtime.h>

namespace ctqw::sparse {

    struct EdgeCsrMap {
        int off_uv;
        int off_vu;
        int diag_u;
        int diag_v;
    };

    class LaplacianWeightUpdater {
        public:
            LaplacianWeightUpdater() = default;

            explicit LaplacianWeightUpdater(
                const std::vector<EdgeCsrMap>& edge_map
            );

            ~LaplacianWeightUpdater();

            LaplacianWeightUpdater(const LaplacianWeightUpdater&) = delete;
            LaplacianWeightUpdater& operator=(const LaplacianWeightUpdater&) = delete;

            LaplacianWeightUpdater(LaplacianWeightUpdater&& other) noexcept;
            LaplacianWeightUpdater& operator=(LaplacianWeightUpdater&& other) noexcept;

            void upload_mapping(
                const std::vector<EdgeCsrMap>& edge_map
            );

            void release();

            void update(
                const float* d_weights,
                std::size_t weight_count,
                cuComplex* d_laplacian_values,
                std::size_t nnz
            ) const;

            std::size_t edge_count() const {
                return edge_count_;
            }

        private:
            EdgeCsrMap* d_edge_map = nullptr;
            std::size_t edge_count_ = 0;
    };

}