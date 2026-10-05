#include "nbody/config.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace nbody {
namespace {

Implementation parse_implementation(std::string_view value) {
    if (value == "serial") return Implementation::serial;
    if (value == "openmp") return Implementation::openmp;
    if (value == "cuda_naive") return Implementation::cuda_naive;
    if (value == "cuda_tiled") return Implementation::cuda_tiled;
    throw std::invalid_argument("unknown implementation: " + std::string(value));
}

const char* require_value(int& index, int argc, char** argv) {
    if (++index >= argc) {
        throw std::invalid_argument("missing value for " + std::string(argv[index - 1]));
    }
    return argv[index];
}

}  // namespace

Config parse_arguments(int argc, char** argv) {
    Config config;
    for (int i = 1; i < argc; ++i) {
        const std::string_view option{argv[i]};
        if (option == "--help" || option == "-h") {
            print_help(argv[0]);
            std::exit(0);
        } else if (option == "--implementation") {
            config.implementation = parse_implementation(require_value(i, argc, argv));
        } else if (option == "--bodies") {
            config.bodies = std::stoull(require_value(i, argc, argv));
        } else if (option == "--steps") {
            config.steps = std::stoull(require_value(i, argc, argv));
        } else if (option == "--dt") {
            config.dt = std::stod(require_value(i, argc, argv));
        } else if (option == "--epsilon") {
            config.epsilon = std::stod(require_value(i, argc, argv));
        } else if (option == "--seed") {
            config.seed = std::stoull(require_value(i, argc, argv));
        } else if (option == "--threads") {
            config.threads = std::stoi(require_value(i, argc, argv));
        } else if (option == "--block-size") {
            config.block_size = std::stoi(require_value(i, argc, argv));
        } else if (option == "--output") {
            config.output = require_value(i, argc, argv);
        } else {
            throw std::invalid_argument("unknown option: " + std::string(option));
        }
    }

    if (config.bodies == 0) throw std::invalid_argument("--bodies must be positive");
    if (config.dt <= 0.0) throw std::invalid_argument("--dt must be positive");
    if (config.epsilon < 0.0) throw std::invalid_argument("--epsilon cannot be negative");
    return config;
}

std::string to_string(Implementation implementation) {
    switch (implementation) {
        case Implementation::serial: return "serial";
        case Implementation::openmp: return "openmp";
        case Implementation::cuda_naive: return "cuda_naive";
        case Implementation::cuda_tiled: return "cuda_tiled";
    }
    return "unknown";
}

void print_help(const char* program_name) {
    std::cout
        << "Usage: " << program_name << " [options]\n\n"
        << "  --implementation serial|openmp|cuda_naive|cuda_tiled\n"
        << "  --bodies N          Number of bodies (default: 1024)\n"
        << "  --steps N           Integration steps (default: 100)\n"
        << "  --dt VALUE          Time step (default: 0.001)\n"
        << "  --epsilon VALUE     Softening value (default: 0.01)\n"
        << "  --seed N            Initialization seed (default: 12345)\n"
        << "  --threads N         OpenMP thread count\n"
        << "  --block-size N      CUDA block size (default: 256)\n"
        << "  --output FILE       Future CSV output path\n";
}

}  // namespace nbody

