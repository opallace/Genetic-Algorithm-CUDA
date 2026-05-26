#include "ga/cuda/core/CudaGeneticAlgorithm.hpp"

#include "ga/cuda/core/CudaOptimization.hpp"
#include "ga/cuda/fitness/CudaTargetFitness.hpp"
#include "ga/cuda/operators/mutation/CudaGaussianMutation.hpp"
#include "ga/cuda/operators/crossover/CudaUniformCrossover.hpp"
#include "ga/cuda/operators/selection/CudaTournamentSelection.hpp"

using namespace ga::cuda::core;
using namespace ga::cuda::mutation;
using namespace ga::cuda::crossover;
using namespace ga::cuda::selection;
using namespace ga::cuda::fitness;

int main() {
    using GA = ga::cuda::core::CudaGeneticAlgorithm<
        float,
        CudaGaussianMutation,
        CudaUniformCrossover,
        CudaTournamentSelection,
        CudaTargetFitness,
        CudaMinimize
    >;

    GA ga(
        1000,
        15,
        CudaGaussianMutation(0.001f),
        CudaUniformCrossover{},
        CudaTournamentSelection(2),
        CudaTargetFitness(0.0f),
        CudaMinimize{}
    );

    ga.initialize_random(
        -10.0f,
        10.0f,
        1234
    );

    ga.run(10000);

    return 0;
}