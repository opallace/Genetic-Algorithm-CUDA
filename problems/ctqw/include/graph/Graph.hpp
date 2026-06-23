#pragma once

#include <vector>
#include <random>
#include <stdexcept>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <string>
#include <iostream>
#include <algorithm>

namespace ctqw::graph {

    struct Edge {
        int u;
        int v;
        float w;
    };

    class Graph {
        public:
            Graph() = default;

            explicit Graph(int vertex_count)
                : vertex_count_(vertex_count)
            {
                validate_vertex_count(vertex_count_);
            }

            int vertex_count() const {
                return vertex_count_;
            }

            std::size_t edge_count() const {
                return edges.size();
            }

            const std::vector<Edge>& get_edges() const {
                return edges;
            }

            void clear() {
                vertex_count_ = 0;
                edges.clear();
            }

            void reset(int vertex_count) {
                validate_vertex_count(vertex_count);

                vertex_count_ = vertex_count;
                edges.clear();
            }

            void add_edge(int u, int v, float w = 1.0f) {
                validate_edge(u, v, w);

                if (u > v) {
                    std::swap(u, v);
                }

                if (has_edge(u, v)) {
                    throw std::invalid_argument(
                        "Graph::add_edge: duplicate edges are not allowed."
                    );
                }

                edges.push_back(Edge{
                    u,
                    v,
                    w
                });
            }

            void generate_erdos_renyi(
                int vertex_count,
                float edge_probability,
                float initial_weight = 1.0f,
                unsigned int seed = 1234
            ) {
                validate_vertex_count(vertex_count);

                if (edge_probability < 0.0f || edge_probability > 1.0f) {
                    throw std::invalid_argument(
                        "Graph::generate_erdos_renyi: edge_probability must be in [0, 1]."
                    );
                }

                if (initial_weight < 0.0f) {
                    throw std::invalid_argument(
                        "Graph::generate_erdos_renyi: initial_weight must be non-negative."
                    );
                }

                reset(vertex_count);

                std::mt19937 rng(seed);
                std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

                for (int u = 0; u < vertex_count_; u++) {
                    for (int v = u + 1; v < vertex_count_; v++) {
                        const float draw = distribution(rng);

                        if (draw <= edge_probability) {
                            add_edge(u, v, initial_weight);
                        }
                    }
                }
            }

            void export_edge_list(const std::string& filename) const
            {
                std::ofstream file(filename);

                if (!file.is_open()) {
                    throw std::runtime_error(
                        "Graph::export_edge_list: could not open file for writing: "
                        + filename
                    );
                }

                file << std::fixed << std::setprecision(8);

                for (const auto& edge : edges) {
                    file << edge.u << " "
                         << edge.v << " "
                         << edge.w << "\n";
                }

                std::cout << "Graph exported to " << filename << std::endl;
            }

            void export_edge_list_with_weights(
                const std::vector<float>& weights,
                const std::string& filename
            ) const {
                if (weights.size() != edges.size()) {
                    throw std::runtime_error(
                        "Graph::export_edge_list_with_weights: number of weights does not match number of edges."
                    );
                }

                std::ofstream file(filename);

                if (!file.is_open()) {
                    throw std::runtime_error(
                        "Graph::export_edge_list_with_weights: could not open file for writing: "
                        + filename
                    );
                }

                file << std::fixed << std::setprecision(8);

                for (std::size_t i = 0; i < edges.size(); i++) {
                    const auto& edge = edges[i];

                    file << edge.u << " "
                         << edge.v << " "
                         << weights[i] << "\n";
                }

                std::cout << "Weighted graph exported to " << filename << std::endl;
            }

        private:
            int vertex_count_ = 0;
            std::vector<Edge> edges;

            static void validate_vertex_count(int vertex_count) {
                if (vertex_count <= 0) {
                    throw std::invalid_argument(
                        "Graph: vertex_count must be positive."
                    );
                }
            }

            void validate_edge(int u, int v, float w) const {
                if (vertex_count_ <= 0) {
                    throw std::invalid_argument(
                        "Graph::add_edge: graph has no vertices."
                    );
                }

                if (u < 0 || u >= vertex_count_) {
                    throw std::invalid_argument(
                        "Graph::add_edge: vertex u is out of range."
                    );
                }

                if (v < 0 || v >= vertex_count_) {
                    throw std::invalid_argument(
                        "Graph::add_edge: vertex v is out of range."
                    );
                }

                if (u == v) {
                    throw std::invalid_argument(
                        "Graph::add_edge: self-loops are not allowed."
                    );
                }

                if (w < 0.0f) {
                    throw std::invalid_argument(
                        "Graph::add_edge: edge weight must be non-negative."
                    );
                }
            }

            bool has_edge(int u, int v) const {
                if (u > v) {
                    std::swap(u, v);
                }

                for (const auto& edge : edges) {
                    if (edge.u == u && edge.v == v) {
                        return true;
                    }
                }

                return false;
            }
    };

}