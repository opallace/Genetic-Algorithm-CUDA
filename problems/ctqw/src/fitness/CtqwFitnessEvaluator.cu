#include "fitness/CtqwFitnessEvaluator.cuh"

#include <stdexcept>

#include <cuda_runtime.h>

#include "utils/CudaCheck.cuh"

namespace ctqw::fitness {

    CtqwFitnessEvaluator::CtqwFitnessEvaluator(
        float norm_penalty,
        ctqw::problem::TransferProblem problem,
        const ctqw::sparse::DeviceWeightedLaplacian& laplacian
    )
        : norm_penalty(norm_penalty),
          problem(problem),
          laplacian(laplacian),
          transfer_evaluator(laplacian)
    {}

    void CtqwFitnessEvaluator::evaluate_population(
        const float* d_population,
        double* d_fitness_values,
        std::size_t population_size,
        std::size_t chromosome_size
        
    ) {
        if (!d_population) {
            throw std::invalid_argument(
                "CtqwFitnessEvaluator: d_population is null."
            );
        }

        if (!d_fitness_values) {
            throw std::invalid_argument(
                "CtqwFitnessEvaluator: d_fitness_values is null."
            );
        }

        if (population_size == 0) {
            throw std::invalid_argument(
                "CtqwFitnessEvaluator: population_size must be greater than zero."
            );
        }

        if (chromosome_size == 0) {
            throw std::invalid_argument(
                "CtqwFitnessEvaluator: chromosome_size must be greater than zero."
            );
        }

        if (chromosome_size != laplacian.edge_count()) {
            throw std::invalid_argument(
                "CtqwFitnessEvaluator: chromosome_size must match the number of graph edges."
            );
        }

        h_fitness_values.resize(population_size);

        for (std::size_t individual = 0; individual < population_size; individual++) {
            const float* d_weights = d_population + individual * chromosome_size;

            h_fitness_values[individual] = evaluate_one_individual(d_weights);
        }

        CTQW_CUDA_CHECK(cudaMemcpy(
            d_fitness_values,
            h_fitness_values.data(),
            sizeof(double) * population_size,
            cudaMemcpyHostToDevice
        ));
    }

    double CtqwFitnessEvaluator::evaluate_one_individual(
        const float* d_weights
    ) {
        auto result = transfer_evaluator.evaluate(problem, d_weights);

        return static_cast<double>(result.target_probability)
            - static_cast<double>(norm_penalty)
            * static_cast<double>(result.norm_error);
    }

}
