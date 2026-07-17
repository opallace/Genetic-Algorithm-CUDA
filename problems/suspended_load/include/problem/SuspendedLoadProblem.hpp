#pragma once

namespace suspended_load::problem {

    /**
     * Physical and numerical configuration for the suspended-load problem.
     *
     * The controlled system is a cart carrying a suspended pendulum load:
     *
     *   x''(t)      = a(t)
     *   theta''(t)  = -(a(t) / l) cos(theta) - (g / l) sin(theta)
     *
     * A chromosome stores a piecewise-constant acceleration profile a_i.
     * Each gene is used during one control interval. The solver can integrate
     * each interval with several RK4 substeps through steps_per_gene.
     */
    struct SuspendedLoadProblem {
        float gravity = 9.80665f;
        float length  = 1.0f;

        float initial_position = 0.0f;
        float target_position  = 1.0f;

        float initial_velocity = 0.0f;
        float initial_theta    = 0.0f;
        float initial_omega    = 0.0f;

        float final_time = 4.0f;
        int steps_per_gene = 4;
    };

    /**
     * Weights used to convert the trajectory into a scalar minimization cost.
     *
     * Large terminal weights force the cart to reach the target and stop.
     * Swing weights reduce the pendulum oscillation at the end and along the
     * trajectory. The control-effort term discourages unnecessarily aggressive
     * accelerations.
     */
    struct SuspendedLoadFitnessWeights {
        float terminal_position = 200.0f;
        float terminal_velocity = 25.0f;
        float terminal_theta    = 250.0f;
        float terminal_omega    = 25.0f;

        float running_theta     = 2.0f;
        float running_omega     = 0.2f;
        float control_effort    = 0.01f;
        float acceleration_smoothness = 0.0f;
    };

    struct SuspendedLoadState {
        float x;
        float v;
        float theta;
        float omega;
    };

} // namespace suspended_load::problem
