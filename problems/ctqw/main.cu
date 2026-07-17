#include <iostream>

#include "ga/core/CudaGeneticAlgorithm.hpp"

#include "ga/core/CudaFitnessComparator.hpp"
#include "ga/population/CudaUniformPopulation.cuh"
#include "ga/population/CudaPopulationManager.hpp"
#include "ga/operators/mutation/CudaPolynomialMutation.hpp"
#include "ga/operators/crossover/CudaSBXCrossover.hpp"
#include "ga/operators/selection/CudaTournamentSelection.hpp"
#include "ga/control/CudaSimpleParameterController.hpp"

#include "graph/Graph.hpp"
#include "sparse/DeviceWeightedLaplacian.cuh"

#include "problem/TransferProblem.hpp"
#include "fitness/CtqwFitnessEvaluator.cuh"

using namespace ga::core;
using namespace ga::mutation;
using namespace ga::crossover;
using namespace ga::selection;
using namespace ga::population;
using namespace ga::control;

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
    const size_t initial_population_size  = 16;
    const size_t max_population_size      = 16;
    const size_t generation_count = 8192;
    const size_t elitism_count    = 4;
    
    // ---------------------------------------------------------------------
    // Gene / weight bounds
    // ---------------------------------------------------------------------
    const float min_edge_weight = 0.0f;
    const float max_edge_weight = 5.0f;

    // ---------------------------------------------------------------------
    // Genetic operators configuration
    // ---------------------------------------------------------------------
    const float crossover_rate   = 0.90f;
    const float crossover_eta    = 15.0f;

    const float mutation_eta     = 20.0f;
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
    // Parameter Control
    // ---------------------------------------------------------------------
    CudaSimpleParameterController parameter_controller;

    const float inv_edge_count =
        1.0f / static_cast<float>(graph.edge_count());

    parameter_controller.max_population_size = max_population_size;

    // Estagnação
    parameter_controller.stagnation_threshold         = 8;
    parameter_controller.strong_stagnation_threshold  = 24;

    parameter_controller.improvement_epsilon = 1.0e-4;
    parameter_controller.strong_improvement_threshold = 0.01f;

    // Mutação:
    // min = ~1 gene mutado por cromossomo
    // max = ~50 genes mutados por cromossomo
    parameter_controller.min_mutation_rate = 1.0f  * inv_edge_count;
    parameter_controller.max_mutation_rate = 50.0f * inv_edge_count;

    // Eta da mutação polinomial:
    // eta baixo  -> saltos grandes
    // eta alto   -> ajustes locais
    parameter_controller.min_mutation_eta = 3.0f;
    parameter_controller.max_mutation_eta = 60.0f;

    // Eta do SBX:
    // eta baixo  -> filhos mais espalhados
    // eta alto   -> filhos próximos dos pais
    parameter_controller.min_crossover_eta = 3.0f;
    parameter_controller.max_crossover_eta = 60.0f;

    // ---------------------------------------------------------------------
    // Weak improvement
    // Melhorou pouco:
    // - reduz levemente mutação
    // - aumenta levemente eta
    // - não muda população
    // ---------------------------------------------------------------------
    parameter_controller.improvement_mutation_rate_factor = 0.995f;
    parameter_controller.improvement_mutation_eta_factor  = 1.003f;
    parameter_controller.improvement_crossover_eta_factor = 1.003f;
    parameter_controller.improvement_population_delta     = 0;

    // ---------------------------------------------------------------------
    // Strong improvement
    // Melhorou bastante:
    // - entra em exploração local
    // - reduz mutação
    // - aumenta eta
    // - NÃO reduz população ainda
    // ---------------------------------------------------------------------
    
    parameter_controller.strong_improvement_mutation_rate_factor = 0.95f;
    parameter_controller.strong_improvement_mutation_eta_factor  = 1.05f;
    parameter_controller.strong_improvement_crossover_eta_factor = 1.03f;
    parameter_controller.strong_improvement_population_delta     = 0;

    // ---------------------------------------------------------------------
    // Weak stagnation
    // Estagnou moderadamente:
    // - aumenta mutação
    // - reduz eta
    // - adiciona poucos indivíduos
    // ---------------------------------------------------------------------
    parameter_controller.stagnation_mutation_rate_factor = 1.20f;
    parameter_controller.stagnation_mutation_eta_factor  = 0.90f;
    parameter_controller.stagnation_crossover_eta_factor = 0.92f;
    parameter_controller.stagnation_population_delta     = 0;

    // ---------------------------------------------------------------------
    // Strong stagnation
    // Estagnou muito:
    // - aumenta mais mutação
    // - reduz mais eta
    // - adiciona mais indivíduos
    // ---------------------------------------------------------------------
    parameter_controller.strong_stagnation_mutation_rate_factor = 1.45f;
    parameter_controller.strong_stagnation_mutation_eta_factor  = 0.75f;
    parameter_controller.strong_stagnation_crossover_eta_factor = 0.80f;
    parameter_controller.strong_stagnation_population_delta     = 0;

    // ---------------------------------------------------------------------
    // CTQW evaluator
    // ---------------------------------------------------------------------
    const size_t chromosome_size = graph.edge_count();

    DeviceWeightedLaplacian device_laplacian(graph);

    CtqwFitnessEvaluator ctqw_evaluator(
        max_population_size,
        norm_penalty,
        transfer_problem,
        device_laplacian
    );

    // ---------------------------------------------------------------------
    // Genetic algorithm
    // ---------------------------------------------------------------------
    CudaPopulationManager<float> population_manager(
        initial_population_size,
        max_population_size,
        chromosome_size
    );

    CudaGeneticAlgorithm<
        float,
        CudaPopulationManager<float>,
        CudaPolynomialMutation,
        CudaSBXCrossover,
        CudaTournamentSelection,
        CudaMaximizeFitness
    > ga(
        elitism_count,
        crossover_rate,
        population_manager,
        CudaPolynomialMutation(
            min_edge_weight,
            max_edge_weight,
            mutation_eta,
            mutation_rate
        ),
        CudaSBXCrossover(
            min_edge_weight,
            max_edge_weight,
            crossover_eta
        ),
        CudaTournamentSelection(tournament_size),
        CudaMaximizeFitness()
    );

    ga.create_population(
        CudaUniformPopulation(
            min_edge_weight,
            max_edge_weight
        ), 
        rng_seed
    );

    ga.run_with(
        generation_count, 
        ctqw_evaluator, 
        parameter_controller, 
        CudaUniformPopulation(
            min_edge_weight,
            max_edge_weight
        ), 
        rng_seed
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