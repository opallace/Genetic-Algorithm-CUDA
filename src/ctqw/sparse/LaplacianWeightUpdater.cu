#include "ctqw/sparse/LaplacianWeightUpdater.cuh"

#include <stdexcept>
#include <utility>

#include "ctqw/utils/CudaCheck.cuh"

namespace ctqw::sparse {

    namespace {

        __global__
        void clear_laplacian_values_kernel(
            cuComplex* values,
            std::size_t value_count
        ) {
            std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;

            if (i >= value_count) {
                return;
            }

            values[i] = make_cuFloatComplex(0.0f, 0.0f);
        }

        __global__
        void update_laplacian_values_kernel(
            const float* weights,
            cuComplex* values,
            const EdgeCsrMap* edge_map,
            std::size_t edge_count
        ) {
            std::size_t e = blockIdx.x * blockDim.x + threadIdx.x;

            if (e >= edge_count) {
                return;
            }

            const float w = weights[e];
            const EdgeCsrMap map = edge_map[e];

            values[map.off_uv] = make_cuFloatComplex(-w, 0.0f);
            values[map.off_vu] = make_cuFloatComplex(-w, 0.0f);

            atomicAdd(&(values[map.diag_u].x), w);
            atomicAdd(&(values[map.diag_v].x), w);
        }

    }

    LaplacianWeightUpdater::LaplacianWeightUpdater(
        const std::vector<EdgeCsrMap>& edge_map
    ) {
        upload_mapping(edge_map);
    }

    LaplacianWeightUpdater::~LaplacianWeightUpdater() {
        release();
    }

    LaplacianWeightUpdater::LaplacianWeightUpdater(
        LaplacianWeightUpdater&& other
    ) noexcept {
        d_edge_map = other.d_edge_map;
        edge_count_ = other.edge_count_;

        other.d_edge_map = nullptr;
        other.edge_count_ = 0;
    }

    LaplacianWeightUpdater& LaplacianWeightUpdater::operator=(
        LaplacianWeightUpdater&& other
    ) noexcept {
        if (this != &other) {
            release();

            d_edge_map = other.d_edge_map;
            edge_count_ = other.edge_count_;

            other.d_edge_map = nullptr;
            other.edge_count_ = 0;
        }

        return *this;
    }

    void LaplacianWeightUpdater::upload_mapping(
        const std::vector<EdgeCsrMap>& edge_map
    ) {
        release();

        edge_count_ = edge_map.size();

        if (edge_count_ == 0) {
            return;
        }

        CTQW_CUDA_CHECK(cudaMalloc(
            &d_edge_map,
            sizeof(EdgeCsrMap) * edge_count_
        ));

        CTQW_CUDA_CHECK(cudaMemcpy(
            d_edge_map,
            edge_map.data(),
            sizeof(EdgeCsrMap) * edge_count_,
            cudaMemcpyHostToDevice
        ));
    }

    void LaplacianWeightUpdater::release() {
        if (d_edge_map) {
            cudaFree(d_edge_map);
            d_edge_map = nullptr;
        }

        edge_count_ = 0;
    }

    void LaplacianWeightUpdater::update(
        const float* d_weights,
        std::size_t weight_count,
        cuComplex* d_laplacian_values,
        std::size_t nnz
    ) const {
        if (!d_edge_map) {
            throw std::runtime_error(
                "LaplacianWeightUpdater: edge map was not uploaded."
            );
        }

        if (!d_weights) {
            throw std::invalid_argument(
                "LaplacianWeightUpdater: d_weights is null."
            );
        }

        if (!d_laplacian_values) {
            throw std::invalid_argument(
                "LaplacianWeightUpdater: d_laplacian_values is null."
            );
        }

        if (weight_count != edge_count_) {
            throw std::invalid_argument(
                "LaplacianWeightUpdater: weight_count must match edge_count."
            );
        }

        constexpr int threads = 256;

        int clear_blocks = static_cast<int>(
            (nnz + threads - 1) / threads
        );

        clear_laplacian_values_kernel<<<clear_blocks, threads>>>(
            d_laplacian_values,
            nnz
        );

        CTQW_CUDA_CHECK(cudaGetLastError());

        int update_blocks = static_cast<int>(
            (edge_count_ + threads - 1) / threads
        );

        update_laplacian_values_kernel<<<update_blocks, threads>>>(
            d_weights,
            d_laplacian_values,
            d_edge_map,
            edge_count_
        );

        CTQW_CUDA_CHECK(cudaGetLastError());
    }

}