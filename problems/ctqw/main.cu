#include <iostream>

#include "ga/core/CudaGeneticAlgorithm.hpp"

#include "ga/core/CudaFitnessComparator.hpp"
#include "ga/operators/mutation/CudaGaussianMutation.hpp"
#include "ga/operators/crossover/CudaUniformCrossover.hpp"
#include "ga/operators/selection/CudaTournamentSelection.hpp"

#include "graph/Graph.hpp"
#include "sparse/DeviceWeightedLaplacian.cuh"

#include "problem/TransferProblem.hpp"
#include "fitness/CtqwFitnessEvaluator.cuh"

using namespace ga::core;
using namespace ga::mutation;
using namespace ga::crossover;
using namespace ga::selection;

using namespace ctqw::graph;
using namespace ctqw::sparse;
using namespace ctqw::problem;
using namespace ctqw::fitness;

using namespace std;

int main() {
    // ---------------------------------------------------------------------
    // Reproducibility
    // ---------------------------------------------------------------------
    const unsigned int rng_seed = 2324;

    // ---------------------------------------------------------------------
    // Graph configuration
    // ---------------------------------------------------------------------
    const int vertex_count          = 100;
    const float edge_probability    = 0.12f;
    const float initial_edge_weight = 1.0f;

    const string input_graph_filename  = "input_graph.txt";
    const string output_graph_filename = "output_graph.txt";

    // ---------------------------------------------------------------------
    // CTQW transfer problem
    // ---------------------------------------------------------------------
    TransferProblem transfer_problem;

    transfer_problem.source     = 0;
    transfer_problem.target     = vertex_count - 1;
    transfer_problem.final_time = 5.0f;
    transfer_problem.steps      = 500;

    // ---------------------------------------------------------------------
    // Fitness configuration
    // ---------------------------------------------------------------------
    const float norm_penalty = 1.0f;

    // ---------------------------------------------------------------------
    // Genetic algorithm configuration
    // ---------------------------------------------------------------------
    const size_t population_size  = 16;
    const size_t generation_count = 512;
    const size_t elitism_count    = 1;

    // ---------------------------------------------------------------------
    // Gene / weight bounds
    // ---------------------------------------------------------------------
    const float min_edge_weight = 0.0f;
    const float max_edge_weight = 2.0f;

    // ---------------------------------------------------------------------
    // Genetic operators configuration
    // ---------------------------------------------------------------------
    const float mutation_sigma   = 0.085f;
    const float mutation_rate    = 0.01f;
    const size_t tournament_size = 2;

    // ---------------------------------------------------------------------
    // Build graph
    // ---------------------------------------------------------------------
    Graph graph;

    graph.generate_erdos_renyi(
        vertex_count,
        edge_probability,
        initial_edge_weight,
        rng_seed
    );

    graph.export_edge_list(input_graph_filename);

    cout << "Number of vertices = " << graph.vertex_count() << "\n";
    cout << "Number of edges    = " << graph.edge_count() << "\n";

    // ---------------------------------------------------------------------
    // CTQW evaluator
    // ---------------------------------------------------------------------
    const size_t chromosome_size = graph.edge_count();

    DeviceWeightedLaplacian device_laplacian(graph);

    CtqwFitnessEvaluator ctqw_evaluator(
        norm_penalty,
        transfer_problem,
        device_laplacian
    );

    // ---------------------------------------------------------------------
    // Genetic algorithm
    // ---------------------------------------------------------------------
    CudaGeneticAlgorithm<
        float,
        CudaGaussianMutation,
        CudaUniformCrossover,
        CudaTournamentSelection,
        CudaMaximizeFitness
    > ga(
        population_size,
        chromosome_size,
        elitism_count,
        CudaGaussianMutation(
            min_edge_weight,
            max_edge_weight,
            mutation_sigma,
            mutation_rate
        ),
        CudaUniformCrossover{},
        CudaTournamentSelection(tournament_size),
        CudaMaximizeFitness{}
    );

    ga.create_uniform_population(
        min_edge_weight,
        max_edge_weight,
        rng_seed
    );

    ga.run_with(
        generation_count,
        ctqw_evaluator
    );

    // ---------------------------------------------------------------------
    // Export result
    // ---------------------------------------------------------------------
    graph.export_edge_list_with_weights(
        ga.best_chromosome(),
        output_graph_filename
    );

    cout << "Best fitness    = " << ga.best_fitness() << "\n";
    cout << "Best generation = " << ga.best_generation() << "\n";

    return 0;
}