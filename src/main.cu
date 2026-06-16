#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdexcept>

#include "ga/core/CudaGeneticAlgorithm.hpp"

#include "ga/core/CudaOptimization.hpp"
#include "ga/operators/mutation/CudaGaussianMutation.hpp"
#include "ga/operators/crossover/CudaUniformCrossover.hpp"
#include "ga/operators/selection/CudaTournamentSelection.hpp"

#include "ctqw/graph/RandomGraph.hpp"
#include "ctqw/sparse/LaplacianBuilder.hpp"
#include "ctqw/sparse/LaplacianStructure.hpp"
#include "ctqw/sparse/DeviceCsrMatrix.cuh"
#include "ctqw/sparse/LaplacianWeightUpdater.cuh"

#include "ctqw/problem/TransferProblem.hpp"
#include "ctqw/fitness/CtqwFitnessEvaluator.cuh"

using namespace ga::core;
using namespace ga::mutation;
using namespace ga::crossover;
using namespace ga::selection;

using namespace ctqw::graph;
using namespace ctqw::sparse;
using namespace ctqw::problem;
using namespace ctqw::fitness;

using namespace std;

void export_best_individual_as_edge_list(
    const ctqw::graph::Graph& graph,
    const std::vector<float>& best_weights,
    const std::string& filename

) {
    if (best_weights.size() != graph.edges.size()) {
        throw std::runtime_error(
            "export_best_individual_as_edge_list: number of weights does not match number of edges."
        );
    }

    std::ofstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error(
            "Could not open file for writing: " + filename
        );
    }

    file << std::fixed << std::setprecision(8);

    for (std::size_t i = 0; i < graph.edges.size(); i++) {
        const auto& edge = graph.edges[i];

        file << edge.u << " "
             << edge.v << " "
             << best_weights[i] << "\n";
    }
}

int main() {
    int n = 100;
    float edge_probability = 0.06f;
    unsigned int seed = 2324;

    auto graph = make_erdos_renyi_graph(n, edge_probability, 1.0f, seed);

    cout << "n = "     << graph.vertex_count() << endl;
    cout << "edges = " << graph.edge_count() << endl;

    auto host_laplacian = LaplacianBuilder::build_complex_csr(graph);

    cout << "nnz = " << host_laplacian.nnz << endl;

    auto edge_map = LaplacianStructure::build_edge_map(graph, host_laplacian);
        

    DeviceCsrMatrix device_laplacian(host_laplacian);
    LaplacianWeightUpdater laplacian_updater(edge_map);
    TransferProblem problem;

    problem.source     = 0;
    problem.target     = n - 1;
    problem.final_time = 5.0f;
    problem.steps      = 1000;

    float norm_penalty = 1.0f;

    CtqwFitnessEvaluator ctqw_evaluator(
        norm_penalty,
        problem,
        device_laplacian,
        laplacian_updater
    );

    const std::size_t population_size = 32;
    const std::size_t chromosome_size = static_cast<std::size_t>(graph.edge_count());

    CudaGeneticAlgorithm<
        float,
        CudaGaussianMutation,
        CudaUniformCrossover,
        CudaTournamentSelection,
        CudaMaximize
    > ga(
        population_size,
        chromosome_size,
        CudaGaussianMutation(0.100f, 0.0005f, 0.010f),
        CudaUniformCrossover{},
        CudaTournamentSelection(2),
        CudaMaximize{}
    );

    ga.initialize_random(0.0f, 2.0f, seed);

    ga.run_with(127, ctqw_evaluator);

    const auto& best_weights = ga.best_chromosome();

    export_best_individual_as_edge_list(graph, best_weights, "output_ga.txt");

    std::cout << "Best individual exported to output_ga.txt\n";
    std::cout << "Best fitness = "    << ga.best_fitness() << "\n";
    std::cout << "Best generation = " << ga.best_generation() << "\n";


    return 0;
}