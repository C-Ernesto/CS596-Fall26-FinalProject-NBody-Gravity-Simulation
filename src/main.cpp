#include "nbody/config.hpp"
#include "nbody/simulation.hpp"

#include <exception>
#include <iostream>
#include <stdexcept>
#include <chrono>
#include <algorithm>
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
        std::sort(simulation_times.begin(), simulation_times.end());
        const std::size_t middle = simulation_times.size() / 2;

        const double median_time = simulation_times.size() % 2 == 0
                        ? (simulation_times[middle - 1] + simulation_times[middle]) / 2.0
                        : simulation_times[middle];

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
        
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}

