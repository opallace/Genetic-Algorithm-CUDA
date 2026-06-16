#pragma once

#include <random>

#include "ctqw/graph/Graph.hpp"

namespace ctqw::graph {

    inline Graph make_erdos_renyi_graph(
        int n,
        float edge_probability,
        float weight = 1.0f,
        unsigned int seed = 1234
    ) {
        
        Graph graph;
        graph.n = n;

        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> prob(0.0f, 1.0f);

        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (prob(rng) < edge_probability) {
                    graph.edges.push_back({i, j, weight});
                }
            }
        }

        return graph;
    }

}