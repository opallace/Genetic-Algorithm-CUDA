#pragma once

namespace ga::core {

    class CudaMaximizeFitness {
        public:
            __host__ __device__
            bool is_better(double candidate, double current_best) const {
                return candidate > current_best;
            }
    };

    class CudaMinimizeFitness {
        public:
            __host__ __device__
            bool is_better(double candidate, double current_best) const {
                return candidate < current_best;
            }
    };

}