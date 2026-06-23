#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdexcept>

#include "ga/core/CudaGeneticAlgorithm.hpp"

#include "ga/core/CudaFitnessComparator.hpp"
#include "ga/operators/mutation/CudaGaussianMutation.hpp"
#include "ga/operators/crossover/CudaUniformCrossover.hpp"
#include "ga/operators/selection/CudaTournamentSelection.hpp"

using namespace ga::core;
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

    int elite_count = 1;
    const size_t population_size = 128;
    const size_t chromosome_size = 50;

    SphereFitness fitness;

    CudaGeneticAlgorithm<
        float,
        CudaGaussianMutation,
        CudaUniformCrossover,
        CudaTournamentSelection,
        CudaMinimizeFitness
    > ga(
        population_size,
        chromosome_size,
        elite_count,
        CudaGaussianMutation(-5.12f, 5.12f, 0.1f, 0.01f),
        CudaUniformCrossover{},
        CudaTournamentSelection(2),
        CudaMinimizeFitness{}
    );
    
    ga.create_uniform_population(-5.12f, 5.12f, seed);
    ga.run<SphereFitness>(500, fitness);

    cout << "Best fitness = "    << ga.best_fitness() << "\n";
    cout << "Best generation = " << ga.best_generation() << "\n";


    return 0;
}