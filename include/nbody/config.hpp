#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace nbody {

enum class Implementation { serial, openmp, cuda_naive, cuda_tiled };

struct Config {
    Implementation implementation{Implementation::serial};
    std::size_t bodies{1024};
    std::size_t steps{100};
    double dt{0.001};
    double epsilon{0.01};
    std::uint64_t seed{12345};
    int threads{0};
    int block_size{256};
    std::string output{};
};

Config parse_arguments(int argc, char** argv);
std::string to_string(Implementation implementation);
void print_help(const char* program_name);

}  // namespace nbody

