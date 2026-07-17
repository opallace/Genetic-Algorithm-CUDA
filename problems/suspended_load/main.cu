#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <sstream>
#include <algorithm>

#include "ga/core/CudaGeneticAlgorithm.hpp"
#include "ga/core/CudaFitnessComparator.hpp"
#include "ga/population/CudaPopulationManager.hpp"
#include "ga/population/CudaUniformPopulation.cuh"
#include "ga/operators/mutation/CudaPolynomialMutation.hpp"
#include "ga/operators/crossover/CudaSBXCrossover.hpp"
#include "ga/operators/selection/CudaTournamentSelection.hpp"
#include "ga/control/CudaSimpleParameterController.hpp"

#include "problem/SuspendedLoadProblem.hpp"
#include "fitness/SuspendedLoadFitnessEvaluator.cuh"

using namespace ga::core;
using namespace ga::population;
using namespace ga::mutation;
using namespace ga::crossover;
using namespace ga::selection;
using namespace ga::control;

using namespace suspended_load::problem;
using namespace suspended_load::fitness;

using namespace std;

namespace {

    string format_number_br(float value) {
        ostringstream oss;
        oss << fixed << setprecision(8) << value;

        string s = oss.str();
        replace(s.begin(), s.end(), '.', ',');

        return s;
    }

    struct CostBreakdown {
        SuspendedLoadState final_state{};
        double terminal_position = 0.0;
        double terminal_velocity = 0.0;
        double terminal_theta = 0.0;
        double terminal_omega = 0.0;
        double running_theta = 0.0;
        double running_omega = 0.0;
        double control_effort = 0.0;
        double total = 0.0;
    };

    SuspendedLoadState add_scaled(
        SuspendedLoadState state,
        SuspendedLoadState derivative,
        float scale
    ) {
        return SuspendedLoadState{
            state.x     + scale * derivative.x,
            state.v     + scale * derivative.v,
            state.theta + scale * derivative.theta,
            state.omega + scale * derivative.omega
        };
    }

    SuspendedLoadState derivative(
        SuspendedLoadState state,
        float acceleration,
        const SuspendedLoadProblem& problem
    ) {
        return SuspendedLoadState{
            state.v,
            acceleration,
            state.omega,
            -(acceleration / problem.length) * cosf(state.theta)
                - (problem.gravity / problem.length) * sinf(state.theta)
        };
    }

    SuspendedLoadState rk4_step(
        SuspendedLoadState state,
        float acceleration,
        float dt,
        const SuspendedLoadProblem& problem
    ) {
        const SuspendedLoadState k1 = derivative(state, acceleration, problem);
        const SuspendedLoadState k2 = derivative(add_scaled(state, k1, 0.5f * dt), acceleration, problem);
        const SuspendedLoadState k3 = derivative(add_scaled(state, k2, 0.5f * dt), acceleration, problem);
        const SuspendedLoadState k4 = derivative(add_scaled(state, k3, dt), acceleration, problem);

        state.x += (dt / 6.0f) * (k1.x + 2.0f * k2.x + 2.0f * k3.x + k4.x);
        state.v += (dt / 6.0f) * (k1.v + 2.0f * k2.v + 2.0f * k3.v + k4.v);
        state.theta += (dt / 6.0f) * (k1.theta + 2.0f * k2.theta + 2.0f * k3.theta + k4.theta);
        state.omega += (dt / 6.0f) * (k1.omega + 2.0f * k2.omega + 2.0f * k3.omega + k4.omega);

        return state;
    }

    void export_oscillation(
        const string& filename,
        const vector<float>& accelerations,
        const SuspendedLoadProblem& problem
    ) {
        ofstream file(filename);

        if (!file) {
            throw runtime_error("Could not open output file: " + filename);
        }

        file << fixed << setprecision(8);

        const int total_steps =
            static_cast<int>(accelerations.size()) * problem.steps_per_gene;

        const float dt = problem.final_time / static_cast<float>(total_steps);

        SuspendedLoadState state{
            problem.initial_position,
            problem.initial_velocity,
            problem.initial_theta,
            problem.initial_omega
        };

        file << "time;x;theta\n";
        file << format_number_br(0.0f) << ";"
             << format_number_br(state.x) << ";"
             << format_number_br(state.theta) << "\n";

        int step = 0;

        for (std::size_t gene = 0; gene < accelerations.size(); gene++) {
            const float acceleration = accelerations[gene];

            for (int substep = 0; substep < problem.steps_per_gene; substep++) {
                state = rk4_step(state, acceleration, dt, problem);
                step++;

                const float time = static_cast<float>(step) * dt;

                file << format_number_br(time) << ";"
                     << format_number_br(state.x) << ";"
                     << format_number_br(state.theta) << "\n";
            }
        }
    }

} // namespace

int main() {
    // ---------------------------------------------------------------------
    // Reproducibility
    // ---------------------------------------------------------------------
    const unsigned int rng_seed = 2324;

    // ---------------------------------------------------------------------
    // Suspended-load physical problem
    // ---------------------------------------------------------------------
    SuspendedLoadProblem problem;

    problem.gravity = 9.80665f;
    problem.length = 1.0f;

    problem.initial_position = 0.0f;
    problem.target_position = 5.0f;

    problem.initial_velocity = 0.0f;
    problem.initial_theta = 1.0f;
    problem.initial_omega = 0.0f;

    problem.final_time = 10.0f;
    problem.steps_per_gene = 4;

    // ---------------------------------------------------------------------
    // Fitness weights
    // ---------------------------------------------------------------------
    SuspendedLoadFitnessWeights weights;

    weights.terminal_position = 100.0f;
    weights.terminal_velocity = 100.0f;
    weights.terminal_theta    = 1000.0f;
    weights.terminal_omega    = 300.0f;
    weights.running_theta     = 1000.0f;
    weights.running_omega     = 20.0f;
    weights.control_effort    = 0.05f;
    weights.acceleration_smoothness = 0.0f;

    // ---------------------------------------------------------------------
    // Genetic encoding
    // ---------------------------------------------------------------------
    const size_t control_intervals = 64;
    const size_t chromosome_size = control_intervals;

    const float min_acceleration = -5.0f;
    const float max_acceleration =  5.0f;

    // ---------------------------------------------------------------------
    // Genetic algorithm configuration
    // ---------------------------------------------------------------------
    const size_t initial_population_size = 64;
    const size_t max_population_size = 128;
    const size_t generation_count = 2048;
    const size_t elitism_count = 4;

    const float crossover_rate = 0.90f;
    const float crossover_eta = 15.0f;

    const float mutation_eta = 20.0f;
    const float mutation_rate = 1.0f / static_cast<float>(chromosome_size);

    const size_t tournament_size = 2;

    cout << "Suspended-load control problem\n";
    cout << "Target position      = " << problem.target_position << " m\n";
    cout << "Final time           = " << problem.final_time << " s\n";
    cout << "Pendulum length      = " << problem.length << " m\n";
    cout << "Control intervals    = " << control_intervals << "\n";
    cout << "RK4 steps per gene   = " << problem.steps_per_gene << "\n";
    cout << "Acceleration bounds  = [" << min_acceleration << ", " << max_acceleration << "] m/s^2\n";

    // ---------------------------------------------------------------------
    // Parameter control
    // ---------------------------------------------------------------------
    CudaSimpleParameterController parameter_controller;

    parameter_controller.min_population_size = initial_population_size;
    parameter_controller.max_population_size = max_population_size;

    parameter_controller.stagnation_threshold = 12;
    parameter_controller.strong_stagnation_threshold = 36;
    parameter_controller.improvement_epsilon = 1.0e-6;
    parameter_controller.strong_improvement_threshold = 0.10f;

    parameter_controller.min_mutation_rate = 0.25f / static_cast<float>(chromosome_size);
    parameter_controller.max_mutation_rate = 8.0f / static_cast<float>(chromosome_size);

    parameter_controller.min_mutation_eta = 3.0f;
    parameter_controller.max_mutation_eta = 80.0f;

    parameter_controller.min_crossover_eta = 3.0f;
    parameter_controller.max_crossover_eta = 80.0f;

    parameter_controller.improvement_mutation_rate_factor = 0.995f;
    parameter_controller.improvement_mutation_eta_factor = 1.003f;
    parameter_controller.improvement_crossover_eta_factor = 1.003f;
    parameter_controller.improvement_population_delta = 0;

    parameter_controller.strong_improvement_mutation_rate_factor = 0.95f;
    parameter_controller.strong_improvement_mutation_eta_factor = 1.04f;
    parameter_controller.strong_improvement_crossover_eta_factor = 1.03f;
    parameter_controller.strong_improvement_population_delta = 0;

    parameter_controller.stagnation_mutation_rate_factor = 1.20f;
    parameter_controller.stagnation_mutation_eta_factor = 0.90f;
    parameter_controller.stagnation_crossover_eta_factor = 0.92f;
    parameter_controller.stagnation_population_delta = 0;

    parameter_controller.strong_stagnation_mutation_rate_factor = 1.45f;
    parameter_controller.strong_stagnation_mutation_eta_factor = 0.75f;
    parameter_controller.strong_stagnation_crossover_eta_factor = 0.80f;
    parameter_controller.strong_stagnation_population_delta = 0;

    // ---------------------------------------------------------------------
    // Evaluator and GA
    // ---------------------------------------------------------------------
    SuspendedLoadFitnessEvaluator evaluator(problem, weights);

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
        CudaMinimizeFitness
    > ga(
        elitism_count,
        crossover_rate,
        population_manager,
        CudaPolynomialMutation(
            min_acceleration,
            max_acceleration,
            mutation_eta,
            mutation_rate
        ),
        CudaSBXCrossover(
            min_acceleration,
            max_acceleration,
            crossover_eta
        ),
        CudaTournamentSelection(tournament_size),
        CudaMinimizeFitness()
    );

    ga.create_population(
        CudaUniformPopulation(
            min_acceleration,
            max_acceleration
        ),
        rng_seed
    );

    ga.run_with(
        generation_count,
        evaluator,
        parameter_controller,
        CudaUniformPopulation(
            min_acceleration,
            max_acceleration
        ),
        rng_seed
    );

    const vector<float>& best_control = ga.best_chromosome();

    export_oscillation(
        "suspended_load_oscillation.csv",
        best_control,
        problem
    );

    return 0;
}