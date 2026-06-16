#pragma once

#include <concepts>
#include <cstddef>

namespace ga::concepts {

    template<typename Selection>
    concept CudaSelectionConcept =
        requires(Selection selection){
            { selection.tournament_size } -> std::convertible_to<std::size_t>;
            
        };

}