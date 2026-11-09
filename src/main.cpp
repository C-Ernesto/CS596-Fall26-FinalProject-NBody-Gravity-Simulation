#include "nbody/config.hpp"
#include "nbody/simulation.hpp"

#include <algorithm>
#include <chrono>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv) {
    try {
        const nbody::Config config = nbody::parse_arguments(argc, argv);
        const nbody::Bodies bodies = nbody::initialize_bodies(config.bodies, config.seed);

        nbody::Bodies final_bodies;

        // lambda function to run simulation
        auto run_simulation = [&](std::size_t steps) -> nbody::Bodies {
            // switch between CPU or CUDA
            switch (config.implementation) {
                case nbody::Implementation::cuda_naive : {
                    #ifdef NBODY_HAS_CUDA
                        // call simulate_cuda_naive and assign to final bodies
                        return nbody::simulate_cuda_naive(bodies, steps, config.dt, config.epsilon, config.block_size);
                    #else
                        throw std::runtime_error("CUDA support is not enabled");
                    #endif
                }
                case nbody::Implementation::cuda_tiled : {
                    #ifdef NBODY_HAS_CUDA
                        // call simulate_cuda_tiled and assign to final bodies
                        return nbody::simulate_cuda_tiled(bodies, steps, config.dt, config.epsilon, config.block_size);
                    #else
                        throw std::runtime_error("CUDA support is not enabled");
                    #endif
                }
                case nbody::Implementation::serial:
                case nbody::Implementation::openmp:
                    throw std::runtime_error("CPU simulation is not implemented yet");
            }
            throw std::runtime_error("Invalid simulation implementation");
        };

        // run a short CUDA warmup
        (void)run_simulation(1);

        // get number of repetitions
        const std::size_t repetitions = config.repetitions;

        std::vector<double> simulation_times;
        simulation_times.reserve(repetitions);

        // run simulation n times
        for (std::size_t i = 0; i < repetitions; i++) {
            // create chrono variable to measure time
            auto simulation_start = std::chrono::steady_clock::now();
            // call simulation
            final_bodies = run_simulation(config.steps);
            // simulation end time
            auto simulation_end = std::chrono::steady_clock::now();
            // get simulation time
            std::chrono::duration<double, std::milli> elapsed_time = simulation_end - simulation_start;
            // add time to vector
            simulation_times.push_back(elapsed_time.count());
        }
        
        // get median time
        std::vector<double> sorted_times = simulation_times;
        std::sort(sorted_times.begin(), sorted_times.end());
        const std::size_t middle = sorted_times.size() / 2;

        const double median_time = sorted_times.size() % 2 == 0
                        ? (sorted_times[middle - 1] + sorted_times[middle]) / 2.0
                        : sorted_times[middle];

        std::cout << "N-body simulation completed\n"
                  << "  implementation: " << nbody::to_string(config.implementation) << '\n'
                  << "  bodies: " << bodies.size() << '\n'
                  << "  steps: " << config.steps << '\n'
                  << "  seed: " << config.seed << '\n'
                  << "  repetitions: " << repetitions << '\n'
                  << "  median time: " << median_time << " ms\n";
        
        // check final_bodies is not empty
        if (!final_bodies.empty()) {
            std::cout << "  first final body position: ("
                    << final_bodies[0].x << ", "
                    << final_bodies[0].y << ", "
                    << final_bodies[0].z << ")\n";
        }

        if (!config.output.empty()) {
            bool write_header = true;
            {
                std::ifstream existing_output(
                    config.output,
                    std::ios::binary | std::ios::ate
                );
                if (existing_output) {
                    write_header = existing_output.tellg() == 0;
                }
            }

            std::ofstream csv_output(config.output, std::ios::app);
            if (!csv_output) {
                throw std::runtime_error(
                    "could not open CSV output file: " + config.output
                );
            }

            csv_output << std::setprecision(17);
            if (write_header) {
                csv_output
                    << "implementation,bodies,steps,dt,epsilon,seed,threads,"
                    << "block_size,repetition,time_ms\n";
            }

            for (std::size_t i = 0; i < simulation_times.size(); ++i) {
                csv_output
                    << nbody::to_string(config.implementation) << ','
                    << config.bodies << ','
                    << config.steps << ','
                    << config.dt << ','
                    << config.epsilon << ','
                    << config.seed << ','
                    << config.threads << ','
                    << config.block_size << ','
                    << (i + 1) << ','
                    << simulation_times[i] << '\n';
            }

            if (!csv_output) {
                throw std::runtime_error(
                    "failed while writing CSV output file: " + config.output
                );
            }

            std::cout << "  results appended to: " << config.output << '\n';
        }
        
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}

