#pragma once

#include <concepts>

namespace ga::concepts {

    template<typename FitnessComparator, typename FitnessType>
    concept CudaFitnessComparatorConcept =
        requires(FitnessComparator fitness_comparator, FitnessType candidate, FitnessType current_best){
            { fitness_comparator.is_better(candidate, current_best) } -> std::convertible_to<bool>;
            
        };

}