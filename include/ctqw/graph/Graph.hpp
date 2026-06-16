#pragma once

#include <vector>

#include "ctqw/graph/Edge.hpp"

namespace ctqw::graph {

    struct Graph {
        int n = 0;
        std::vector<Edge> edges;

        int vertex_count() const {
            return n;
        }

        int edge_count() const {
            return static_cast<int>(edges.size());
        }
    };

}