#pragma once

#include <utility>
#include <algorithm>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <random>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "ga/utils/CudaCheck.cuh"

#include "ga/concepts/CudaPopulationInitializerConcept.hpp"

#include "ga/kernels/CudaRandomKernels.cuh"
#include "ga/kernels/CudaPopulationManagerKernels.cuh"

namespace ga::population
{

    template<typename GeneType>
    class CudaPopulationManager
    {
        public:

            CudaPopulationManager(
                std::size_t initial_population_size_,
                std::size_t max_population_size_,
                std::size_t chromosome_size_
            )
                : active_population_size(initial_population_size_),
                  max_population_size(max_population_size_),
                  chromosome_size(chromosome_size_),
                  max_total_genes(max_population_size_ * chromosome_size_)
            {
                validate();
                allocate();
            }

            ~CudaPopulationManager()
            {
                release();
            }

            CudaPopulationManager(const CudaPopulationManager&) = delete;
            CudaPopulationManager& operator=(const CudaPopulationManager&) = delete;

            CudaPopulationManager(CudaPopulationManager&&) = delete;
            CudaPopulationManager& operator=(CudaPopulationManager&&) = delete;

            GeneType* data()
            {
                return d_population;
            }

            const GeneType* data() const
            {
                return d_population;
            }

            GeneType* next_data()
            {
                return d_next_population;
            }

            const GeneType* next_data() const
            {
                return d_next_population;
            }

            curandState* gene_rng_states()
            {
                return d_gene_rng_states;
            }

            const curandState* gene_rng_states() const
            {
                return d_gene_rng_states;
            }

            std::size_t size() const
            {
                return active_population_size;
            }

            std::size_t capacity() const
            {
                return max_population_size;
            }

            std::size_t chromosome_length() const
            {
                return chromosome_size;
            }

            std::size_t active_gene_count() const
            {
                return active_population_size * chromosome_size;
            }

            std::size_t max_gene_count() const
            {
                return max_total_genes;
            }

            void swap_buffers()
            {
                std::swap(d_population, d_next_population);
            }

            template<typename PopulationInitializer>
            requires ga::concepts::CudaPopulationInitializerConcept<
                PopulationInitializer,
                GeneType
            >
            void initialize(
                PopulationInitializer initializer,
                unsigned long seed = 1234
            )
            {
                ga::kernels::launch_setup_rng(
                    d_gene_rng_states,
                    max_total_genes,
                    seed
                );

                ga::kernels::launch_initialize_population(
                    d_population,
                    d_gene_rng_states,
                    active_gene_count(),
                    initializer
                );
            }

            template<typename PopulationInitializer>
            requires ga::concepts::CudaPopulationInitializerConcept<
                PopulationInitializer,
                GeneType
            >
            void add_individuals(
                std::size_t count,
                PopulationInitializer initializer,
                unsigned long seed = 1234
            )
            {
                if (count == 0){
                    return;
                }

                const std::size_t new_population_size = active_population_size + count;

                if (new_population_size > max_population_size){
                    throw std::invalid_argument(
                        "Cannot add individuals: population capacity exceeded."
                    );
                }

                if (new_population_size % 2 != 0){
                    throw std::invalid_argument(
                        "Active population size must remain even."
                    );
                }

                const std::size_t offset_genes = active_population_size * chromosome_size;
                const std::size_t added_genes  = count * chromosome_size;

                ga::kernels::launch_setup_rng(
                    d_gene_rng_states + offset_genes,
                    added_genes,
                    seed
                );

                ga::kernels::launch_initialize_population(
                    d_population + offset_genes,
                    d_gene_rng_states + offset_genes,
                    added_genes,
                    initializer
                );

                active_population_size = new_population_size;
            }

            void kill_random_individuals(
                std::size_t count,
                std::size_t protected_tail_count,
                unsigned long seed = 1234
            )
            {
                if (count == 0){
                    return;
                }

                if (count >= active_population_size){
                    throw std::invalid_argument(
                        "Cannot kill all individuals."
                    );
                }

                if (protected_tail_count >= active_population_size){
                    throw std::invalid_argument(
                        "protected_tail_count must be smaller than population size."
                    );
                }

                const std::size_t new_population_size = active_population_size - count;

                if (new_population_size < 2){
                    throw std::invalid_argument(
                        "Population size must remain at least 2."
                    );
                }

                if (new_population_size % 2 != 0){
                    throw std::invalid_argument(
                        "Active population size must remain even."
                    );
                }

                if (new_population_size <= protected_tail_count){
                    throw std::invalid_argument(
                        "New population size must be larger than protected tail count."
                    );
                }

                const std::size_t random_region_size    = active_population_size - protected_tail_count;
                const std::size_t random_survivor_count = new_population_size - protected_tail_count;

                std::vector<int> candidate_indices(random_region_size);

                std::iota(
                    candidate_indices.begin(),
                    candidate_indices.end(),
                    0
                );

                std::mt19937 rng(seed);

                std::shuffle(candidate_indices.begin(), candidate_indices.end(), rng);

                std::vector<int> selected_indices;

                selected_indices.reserve(new_population_size);

                for (std::size_t i = 0; i < random_survivor_count; i++){
                    selected_indices.push_back(candidate_indices[i]);
                }

                const std::size_t protected_begin = active_population_size - protected_tail_count;

                for (std::size_t i = protected_begin; i < active_population_size; i++){
                    selected_indices.push_back(static_cast<int>(i));
                }

                GA_CUDA_CHECK(cudaMemcpy(
                    d_selected_indices,
                    selected_indices.data(),
                    new_population_size * sizeof(int),
                    cudaMemcpyHostToDevice
                ));

                ga::kernels::launch_copy_selected_individuals(
                    d_population,
                    d_next_population,
                    d_selected_indices,
                    new_population_size,
                    chromosome_size
                );

                swap_buffers();

                active_population_size = new_population_size;
            }

            template<typename PopulationInitializer>
            requires ga::concepts::CudaPopulationInitializerConcept<
                PopulationInitializer,
                GeneType
            >
            void resize(
                std::size_t new_population_size,
                PopulationInitializer initializer,
                std::size_t protected_tail_count,
                unsigned long seed = 1234
            )
            {
                if (new_population_size == active_population_size){
                    return;
                }

                if (new_population_size > max_population_size){
                    throw std::invalid_argument(
                        "New population size cannot exceed population capacity."
                    );
                }

                if (new_population_size < 2){
                    throw std::invalid_argument(
                        "Population size must remain at least 2."
                    );
                }

                if (new_population_size % 2 != 0){
                    throw std::invalid_argument(
                        "Active population size must remain even."
                    );
                }

                if (protected_tail_count >= active_population_size){
                    throw std::invalid_argument(
                        "protected_tail_count must be smaller than active population size."
                    );
                }

                if (new_population_size <= protected_tail_count){
                    throw std::invalid_argument(
                        "New population size must be larger than protected tail count."
                    );
                }

                if (new_population_size > active_population_size){
                    add_individuals(
                        new_population_size - active_population_size,
                        initializer,
                        seed
                    );

                    return;
                }

                kill_random_individuals(
                    active_population_size - new_population_size,
                    protected_tail_count,
                    seed
                );
            }

            std::vector<GeneType> copy_to_host() const
            {
                std::vector<GeneType> h_population(active_gene_count());

                GA_CUDA_CHECK(cudaMemcpy(
                    h_population.data(),
                    d_population,
                    active_gene_count() * sizeof(GeneType),
                    cudaMemcpyDeviceToHost
                ));

                return h_population;
            }

        private:

            std::size_t active_population_size;
            std::size_t max_population_size;
            std::size_t chromosome_size;
            std::size_t max_total_genes;

            GeneType* d_population = nullptr;
            GeneType* d_next_population = nullptr;

            curandState* d_gene_rng_states = nullptr;

            int* d_selected_indices = nullptr;

            void validate() const
            {
                if (chromosome_size == 0){
                    throw std::invalid_argument(
                        "Chromosome size must be greater than zero."
                    );
                }

                if (initial_size_invalid()){
                    throw std::invalid_argument(
                        "Initial population size must be even and at least 2."
                    );
                }

                if (max_population_size < active_population_size){
                    throw std::invalid_argument(
                        "Max population size must be greater than or equal to initial population size."
                    );
                }

                if (max_population_size % 2 != 0){
                    throw std::invalid_argument(
                        "Max population size must be even."
                    );
                }
            }

            bool initial_size_invalid() const
            {
                return
                    active_population_size < 2 ||
                    active_population_size % 2 != 0;
            }

            void allocate()
            {
                GA_CUDA_CHECK(cudaMalloc(
                    &d_population,
                    max_total_genes * sizeof(GeneType)
                ));

                GA_CUDA_CHECK(cudaMalloc(
                    &d_next_population,
                    max_total_genes * sizeof(GeneType)
                ));

                GA_CUDA_CHECK(cudaMalloc(
                    &d_gene_rng_states,
                    max_total_genes * sizeof(curandState)
                ));

                GA_CUDA_CHECK(cudaMalloc(
                    &d_selected_indices,
                    max_population_size * sizeof(int)
                ));
            }

            void release()
            {
                if (d_population){
                    cudaFree(d_population);
                }

                if (d_next_population){
                    cudaFree(d_next_population);
                }

                if (d_gene_rng_states){
                    cudaFree(d_gene_rng_states);
                }

                if (d_selected_indices){
                    cudaFree(d_selected_indices);
                }
            }
    };

}