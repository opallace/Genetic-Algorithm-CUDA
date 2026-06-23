#pragma once

#include <cmath>
#include <vector>

#include <cuComplex.h>

namespace ctqw::observable {

    inline float probability_at(
        const std::vector<cuComplex>& psi,
        int vertex
    ) {
        const float re = cuCrealf(psi[vertex]);
        const float im = cuCimagf(psi[vertex]);

        return re * re + im * im;
    }

    inline float total_probability(
        const std::vector<cuComplex>& psi
    ) {
        float total = 0.0f;

        for (const auto& value : psi) {
            const float re = cuCrealf(value);
            const float im = cuCimagf(value);

            total += re * re + im * im;
        }

        return total;
    }

}