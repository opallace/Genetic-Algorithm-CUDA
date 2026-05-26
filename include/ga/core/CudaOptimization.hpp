#pragma once

namespace ga::cuda::core {

    class CudaMaximize {
        public:
            __host__ __device__
            bool is_better(double candidate, double current_best) const {
                return candidate > current_best;
            }
    };

    class CudaMinimize {
        public:
            __host__ __device__
            bool is_better(double candidate, double current_best) const {
                return candidate < current_best;
            }
    };

}