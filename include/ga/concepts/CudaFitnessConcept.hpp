#pragma once

#include <concepts>

namespace ga::cuda::concepts {

    template<typename Fitness, typename Chromosome>
    concept CudaFitnessConcept =
        requires(Fitness fitness, Chromosome chromosome){
            { fitness(chromosome) } -> std::convertible_to<double>;
            
        };

}