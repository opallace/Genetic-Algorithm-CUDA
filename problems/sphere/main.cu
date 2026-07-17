#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdexcept>

#include "ga/core/CudaGeneticAlgorithm.hpp"

#include "ga/core/CudaFitnessComparator.hpp"
#include "ga/population/CudaPopulationManager.hpp"
#include "ga/population/CudaUniformPopulation.cuh"
#include "ga/operators/mutation/CudaGaussianMutation.hpp"
#include "ga/operators/crossover/CudaUniformCrossover.hpp"
#include "ga/operators/selection/CudaTournamentSelection.hpp"

using namespace ga::core;
using namespace ga::population;
using namespace ga::mutation;
using namespace ga::crossover;
using namespace ga::selection;

using namespace std;

class SphereFitness {
public:
    template<typename Chromosome>
    __host__ __device__
    double operator()(Chromosome chromosome) const {
        double fitness = 0.0;

        for (std::size_t locus = 0; locus < chromosome.size(); locus++) {
            const double x = static_cast<double>(chromosome.allele(locus));
            fitness += x * x;
        }

        return fitness;
    }
};

int main() {
    unsigned int seed = 2324;

    int elite_count = 2;
    const size_t initial_population_size = 16;
    const size_t max_population_size     = 500;
    const size_t chromosome_size         = 50;
    const float crossover_rate           = 0.9f;

    SphereFitness fitness;

    CudaPopulationManager<float> population_manager(
        initial_population_size,
        max_population_size,
        chromosome_size
    );

    CudaGeneticAlgorithm<
        float,
        CudaPopulationManager<float>,
        CudaGaussianMutation,
        CudaUniformCrossover,
        CudaTournamentSelection,
        CudaMinimizeFitness
    > ga(
        elite_count,
        crossover_rate,
        population_manager,
        CudaGaussianMutation(-5.12f, 5.12f, 0.1f, 0.01f),
        CudaUniformCrossover(),
        CudaTournamentSelection(2),
        CudaMinimizeFitness()
    );
    
    ga.create_population(
        CudaUniformPopulation(
            -5.12f,
            +5.12f
        ),
        seed
    );

    ga.run<SphereFitness>(500, fitness);

    cout << "Best fitness = "    << ga.best_fitness() << "\n";
    cout << "Best generation = " << ga.best_generation() << "\n";


    return 0;
}