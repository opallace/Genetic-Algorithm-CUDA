#include "ctqw/fitness/CtqwFitnessEvaluator.cuh"

#include <stdexcept>

#include <cuda_runtime.h>

#include "ctqw/utils/CudaCheck.cuh"

namespace ctqw::fitness {

    CtqwFitnessEvaluator::CtqwFitnessEvaluator(
        float norm_penalty,
        ctqw::problem::TransferProblem problem,
        ctqw::sparse::DeviceCsrMatrix& device_laplacian,
        ctqw::sparse::LaplacianWeightUpdater& laplacian_updater
    )
        : norm_penalty(norm_penalty),
        problem(problem),
        device_laplacian(device_laplacian),
        laplacian_updater(laplacian_updater),
        transfer_evaluator(device_laplacian)
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

        if (chromosome_size != laplacian_updater.edge_count()) {
            throw std::invalid_argument(
                "CtqwFitnessEvaluator: chromosome_size must match the number of graph edges."
            );
        }

        h_fitness_values.resize(population_size);

        for (std::size_t individual = 0; individual < population_size; individual++) {
            const float* d_weights =
                d_population + individual * chromosome_size;

            h_fitness_values[individual] = evaluate_one_individual(
                d_weights,
                chromosome_size
            );
        }

        CTQW_CUDA_CHECK(cudaMemcpy(
            d_fitness_values,
            h_fitness_values.data(),
            sizeof(double) * population_size,
            cudaMemcpyHostToDevice
        ));
    }

    double CtqwFitnessEvaluator::evaluate_one_individual(
        const float* d_weights,
        std::size_t chromosome_size
    ) {
        laplacian_updater.update(
            d_weights,
            chromosome_size,
            device_laplacian.values(),
            static_cast<std::size_t>(device_laplacian.nonzeros())
        );

        auto result = transfer_evaluator.evaluate(problem);

        return static_cast<double>(result.target_probability)
            - static_cast<double>(norm_penalty)
            * static_cast<double>(result.norm_error);
    }

}