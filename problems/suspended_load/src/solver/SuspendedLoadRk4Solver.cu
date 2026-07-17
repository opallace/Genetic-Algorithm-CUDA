#include "solver/SuspendedLoadRk4Solver.cuh"

#include <cuda_runtime.h>
#include <math.h>

#include "ga/utils/CudaCheck.cuh"

namespace suspended_load::solver {

    namespace {

        using suspended_load::problem::SuspendedLoadFitnessWeights;
        using suspended_load::problem::SuspendedLoadProblem;
        using suspended_load::problem::SuspendedLoadState;

        __device__
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

        __device__
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

        __device__
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

        __global__
        void evaluate_suspended_load_fitness_kernel(
            const float* population,
            double* fitness_values,
            std::size_t population_size,
            std::size_t chromosome_size,
            SuspendedLoadProblem problem,
            SuspendedLoadFitnessWeights weights
        ) {
            const std::size_t individual = blockIdx.x * blockDim.x + threadIdx.x;

            if (individual >= population_size) {
                return;
            }

            const float* accelerations = population + individual * chromosome_size;

            SuspendedLoadState state{
                problem.initial_position,
                problem.initial_velocity,
                problem.initial_theta,
                problem.initial_omega
            };

            const int steps_per_gene = problem.steps_per_gene;
            const int total_steps    = static_cast<int>(chromosome_size) * steps_per_gene;

            const float dt = problem.final_time / static_cast<float>(total_steps);

            double theta_integral  = 0.0;
            double omega_integral  = 0.0;
            double effort_integral = 0.0;

            for (std::size_t gene = 0; gene < chromosome_size; gene++) {
                const float acceleration = accelerations[gene];

                for (int substep = 0; substep < steps_per_gene; substep++) {
                    state = rk4_step(state, acceleration, dt, problem);

                    theta_integral +=
                        static_cast<double>(state.theta) *
                        static_cast<double>(state.theta) *
                        static_cast<double>(dt);

                    omega_integral +=
                        static_cast<double>(state.omega) *
                        static_cast<double>(state.omega) *
                        static_cast<double>(dt);

                    effort_integral +=
                        static_cast<double>(acceleration) *
                        static_cast<double>(acceleration) *
                        static_cast<double>(dt);
                }
            }

            double smoothness_sum = 0.0;

            for (std::size_t gene = 1; gene < chromosome_size; gene++) {
                const double da =
                    static_cast<double>(accelerations[gene]) -
                    static_cast<double>(accelerations[gene - 1]);

                smoothness_sum += da * da;
            }

            const double position_error =
                static_cast<double>(state.x) -
                static_cast<double>(problem.target_position);

            const double terminal_cost =
                static_cast<double>(weights.terminal_position) *
                    position_error * position_error +

                static_cast<double>(weights.terminal_velocity) *
                    static_cast<double>(state.v) *
                    static_cast<double>(state.v) +

                static_cast<double>(weights.terminal_theta) *
                    static_cast<double>(state.theta) *
                    static_cast<double>(state.theta) +

                static_cast<double>(weights.terminal_omega) *
                    static_cast<double>(state.omega) *
                    static_cast<double>(state.omega);

            const double running_cost =
                static_cast<double>(weights.running_theta) * theta_integral +
                static_cast<double>(weights.running_omega) * omega_integral;

            const double effort_cost =
                static_cast<double>(weights.control_effort) * effort_integral;

            const double smooth_cost =
                static_cast<double>(weights.acceleration_smoothness) *
                smoothness_sum;

        
            fitness_values[individual] =
                terminal_cost +
                running_cost +
                effort_cost +
                smooth_cost;
        }

    } 

    void launch_evaluate_suspended_load_fitness(
        const float* d_population,
        double* d_fitness_values,
        std::size_t population_size,
        std::size_t chromosome_size,
        SuspendedLoadProblem problem,
        SuspendedLoadFitnessWeights weights
    ) {
        constexpr int threads = 256;

        const int blocks = static_cast<int>(
            (population_size + threads - 1) / threads
        );

        evaluate_suspended_load_fitness_kernel<<<blocks, threads>>>(
            d_population,
            d_fitness_values,
            population_size,
            chromosome_size,
            problem,
            weights
        );

        GA_CUDA_CHECK(cudaGetLastError());
        GA_CUDA_CHECK(cudaDeviceSynchronize());
    }

} // namespace suspended_load::solver
