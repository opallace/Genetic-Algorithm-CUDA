#pragma once

#include <cstddef>

namespace ga::core {

    /**
     * @brief Lightweight chromosome view used by CUDA genetic operators and fitness functions.
     *
     * This class represents a chromosome as a contiguous array of genes stored in memory.
     * It does not own the memory pointed to by @ref data; it only provides indexed access
     * to the alleles and stores the chromosome size.
     *
     * The class is marked with `__host__ __device__` methods so it can be used both on
     * the CPU and inside CUDA kernels.
     *
     * @tparam GeneType Type of each gene in the chromosome.
     *
     * @warning This class does not perform bounds checking. Accessing an invalid locus
     * results in undefined behavior.
     *
     * @warning This class is a non-owning view. The memory pointed to by @ref data must
     * remain valid while the chromosome is being used.
     */
    template<typename GeneType>
    class CudaChromosome {
        public:
            /**
             * @brief Type alias for the chromosome gene type.
             */
            using gene_type = GeneType;

            /**
             * @brief Pointer to the first gene of the chromosome.
             *
             * The genes are expected to be stored contiguously in memory.
             * This class does not allocate, deallocate, or manage this memory.
             */
            GeneType* data;

            /**
             * @brief Number of genes in the chromosome.
             */
            std::size_t chromosome_size;

            /**
             * @brief Returns a mutable reference to the allele at the given locus.
             *
             * @param locus Index of the allele to access.
             *
             * @return Mutable reference to the allele at @p locus.
             *
             * @warning No bounds checking is performed.
             */
            __host__ __device__
            GeneType& allele(std::size_t locus) {
                return data[locus];
            }

            /**
             * @brief Returns a const reference to the allele at the given locus.
             *
             * @param locus Index of the allele to access.
             *
             * @return Const reference to the allele at @p locus.
             *
             * @warning No bounds checking is performed.
             */
            __host__ __device__
            const GeneType& allele(std::size_t locus) const {
                return data[locus];
            }

            /**
             * @brief Returns the number of genes in the chromosome.
             *
             * @return Chromosome size.
             */
            __host__ __device__
            std::size_t size() const {
                return chromosome_size;
            }
    };

}