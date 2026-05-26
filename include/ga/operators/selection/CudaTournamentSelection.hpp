#pragma once

#include <cstddef>

namespace ga::cuda::selection {

    class CudaTournamentSelection {
        public:
            std::size_t tournament_size;

            __host__ __device__
            explicit CudaTournamentSelection(std::size_t tournament_size)
                        : tournament_size(tournament_size) {}

    };

}