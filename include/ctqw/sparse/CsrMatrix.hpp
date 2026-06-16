#pragma once

#include <vector>

namespace ctqw::sparse {

    template<typename ValueType>
    struct CsrMatrix {
        int rows = 0;
        int cols = 0;
        int nnz  = 0;

        std::vector<int> row_offsets;
        std::vector<int> col_indices;
        std::vector<ValueType> values;
    };

}