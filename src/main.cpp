#include "nbody/config.hpp"
#include "nbody/simulation.hpp"

#include <exception>
#include <iostream>

int main(int argc, char** argv) {
    try {
        const nbody::Config config = nbody::parse_arguments(argc, argv);
        const nbody::Bodies bodies = nbody::initialize_bodies(config.bodies, config.seed);

        std::cout << "N-body project skeleton ready\n"
                  << "  implementation: " << nbody::to_string(config.implementation) << '\n'
                  << "  bodies: " << bodies.size() << '\n'
                  << "  steps: " << config.steps << '\n'
                  << "  seed: " << config.seed << '\n'
                  << "Simulation kernels are intentionally deferred to the next milestone.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}

