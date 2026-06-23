#pragma once

namespace ga::core {

    /**
     * @brief Fitness comparator for maximization problems.
     *
     * This comparator defines that a candidate fitness value is better
     * when it is greater than the current best fitness value.
     *
     * Use this comparator when the genetic algorithm should maximize
     * the objective function.
     */
    class CudaMaximizeFitness {
        public:
            /**
             * @brief Checks whether a candidate fitness is better than the current best.
             *
             * @param candidate Fitness value of the candidate individual.
             * @param current_best Current best fitness value found so far.
             *
             * @return true if @p candidate is greater than @p current_best.
             * @return false otherwise.
             */
            __host__ __device__
            bool is_better(double candidate, double current_best) const {
                return candidate > current_best;
            }
    };

    /**
     * @brief Fitness comparator for minimization problems.
     *
     * This comparator defines that a candidate fitness value is better
     * when it is smaller than the current best fitness value.
     *
     * Use this comparator when the genetic algorithm should minimize
     * the objective function.
     */
    class CudaMinimizeFitness {
        public:
            /**
             * @brief Checks whether a candidate fitness is better than the current best.
             *
             * @param candidate Fitness value of the candidate individual.
             * @param current_best Current best fitness value found so far.
             *
             * @return true if @p candidate is smaller than @p current_best.
             * @return false otherwise.
             */
            __host__ __device__
            bool is_better(double candidate, double current_best) const {
                return candidate < current_best;
            }
    };

}