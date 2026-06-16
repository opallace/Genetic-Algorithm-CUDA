#pragma once

#include <cstddef>

namespace ga::core {

    template<typename GeneType>
    class CudaChromosome  {
        public:
            using gene_type = GeneType;

            GeneType* data;
            std::size_t chromosome_size;

            __host__ __device__
            GeneType& allele(std::size_t locus) { 
                return data[locus]; 

            }
            
            __host__ __device__
            const GeneType& allele(std::size_t locus) const {
                return data[locus];

            }

            __host__ __device__
            std::size_t size() const {
                return chromosome_size;
            }
    };

}
