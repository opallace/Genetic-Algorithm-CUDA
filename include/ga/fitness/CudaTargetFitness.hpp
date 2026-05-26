#pragma once

#include <cstddef>

namespace ga::cuda::fitness {

class CudaTargetFitness {
public:
    float target;

    __host__ __device__
    explicit CudaTargetFitness(float target)
        : target(target) {}

    template<typename Chromosome>
    __device__
    double operator()(Chromosome chromosome) const {
        double error = 0.0;

        for (std::size_t locus = 0; locus < chromosome.size(); locus++) {
            double x = static_cast<double>(chromosome.allele(locus));
            double diff = x - static_cast<double>(target);

            error += diff * diff;
        }

        return error;
    }
};

}