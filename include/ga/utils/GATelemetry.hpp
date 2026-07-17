#pragma once

#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <string>

namespace ga::utils
{
    class GATelemetry
    {
        public:

            explicit GATelemetry(
                const std::string& filename = "ga_telemetry.csv"
            )
                : file(filename)
            {
                if (!file.is_open())
                {
                    throw std::runtime_error(
                        "Failed to open telemetry CSV file."
                    );
                }

                file
                    << "generation,"
                    << "current_best_fitness,"
                    << "global_best_fitness,"
                    << "mutation_rate,"
                    << "mutation_eta,"
                    << "crossover_eta,"
                    << "population_size\n";

                file.flush();
            }

            void add_record(
                std::size_t generation,
                double current_best_fitness,
                double global_best_fitness,
                float mutation_rate,
                float mutation_eta,
                float crossover_eta,
                std::size_t population_size
            )
            {
                file
                    << generation << ","
                    << current_best_fitness << ","
                    << global_best_fitness << ","
                    << mutation_rate << ","
                    << mutation_eta << ","
                    << crossover_eta << ","
                    << population_size << "\n";

                file.flush();
            }

        private:

            std::ofstream file;
    };
}