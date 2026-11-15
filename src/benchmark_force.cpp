#include "nbody/initial_conditions.hpp"
#include "nbody/simulation.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Options {
    std::vector<std::size_t> body_counts{4096, 8192, 16384};
    std::uint64_t base_seed{600000};
    std::size_t trajectories_per_count{1};
    std::size_t repetitions{5};
    double epsilon{0.01};
    int block_size{64};
    nbody::InitialConditionFamily family{nbody::InitialConditionFamily::uniform_cube};
    std::filesystem::path output{"results/cuda_force_large_n.csv"};
};

const char* require_value(int& index, int argc, char** argv) {
    if (++index >= argc) {
        throw std::invalid_argument("missing value for " + std::string(argv[index - 1]));
    }
    return argv[index];
}

std::size_t positive_size(const std::string& text, const char* option) {
    std::size_t consumed = 0;
    const long long value = std::stoll(text, &consumed);
    if (consumed != text.size() || value <= 0) {
        throw std::invalid_argument(std::string(option) + " must be positive");
    }
    return static_cast<std::size_t>(value);
}

std::vector<std::size_t> body_counts(const std::string& text) {
    std::vector<std::size_t> values;
    std::stringstream stream(text);
    std::string token;
    while (std::getline(stream, token, ',')) {
        values.push_back(positive_size(token, "--body-counts"));
    }
    if (values.empty()) {
        throw std::invalid_argument("--body-counts cannot be empty");
    }
    return values;
}

Options parse_arguments(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view option{argv[index]};
        if (option == "--body-counts") {
            options.body_counts = body_counts(require_value(index, argc, argv));
        } else if (option == "--base-seed") {
            options.base_seed = std::stoull(require_value(index, argc, argv));
        } else if (option == "--repetitions") {
            options.repetitions = positive_size(
                require_value(index, argc, argv), "--repetitions");
        } else if (option == "--trajectories-per-count") {
            options.trajectories_per_count = positive_size(
                require_value(index, argc, argv), "--trajectories-per-count");
        } else if (option == "--epsilon") {
            options.epsilon = std::stod(require_value(index, argc, argv));
        } else if (option == "--block-size") {
            options.block_size = static_cast<int>(positive_size(
                require_value(index, argc, argv), "--block-size"));
        } else if (option == "--initial-condition") {
            options.family = nbody::parse_initial_condition(
                require_value(index, argc, argv));
        } else if (option == "--output") {
            options.output = require_value(index, argc, argv);
        } else if (option == "--help" || option == "-h") {
            std::cout
                << "Usage: " << argv[0] << " [options]\n"
                << "  --body-counts LIST\n"
                << "  --base-seed N\n"
                << "  --repetitions N\n"
                << "  --trajectories-per-count N\n"
                << "  --epsilon VALUE\n"
                << "  --block-size N\n"
                << "  --initial-condition NAME\n"
                << "  --output FILE\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown option: " + std::string(option));
        }
    }
    if (options.epsilon < 0.0 || options.block_size <= 0 || options.output.empty()) {
        throw std::invalid_argument("invalid force benchmark configuration");
    }
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_arguments(argc, argv);
        if (options.output.has_parent_path()) {
            std::filesystem::create_directories(options.output.parent_path());
        }
        std::ofstream output(options.output, std::ios::trunc);
        if (!output) {
            throw std::runtime_error("could not open output: " + options.output.string());
        }
        output << std::setprecision(17)
               << "implementation,initial_condition,bodies,seed,repetition,allocation_ms,h2d_ms,"
                  "kernel_ms,d2h_ms,cleanup_ms,total_ms\n";

        const nbody::InitialConditionParameters parameters{};
        for (std::size_t count_index = 0;
             count_index < options.body_counts.size();
             ++count_index) {
            const std::size_t count = options.body_counts[count_index];
            for (std::size_t trajectory = 0;
                 trajectory < options.trajectories_per_count;
                 ++trajectory) {
                const std::uint64_t seed =
                    options.base_seed +
                    count_index * options.trajectories_per_count + trajectory;
                const nbody::Bodies bodies = nbody::initialize_bodies_for_family(
                    count, seed, options.family, parameters);
                (void)nbody::compute_acceleration_cuda_tiled(
                    bodies, options.epsilon, options.block_size);

                for (std::size_t repetition = 1;
                     repetition <= options.repetitions;
                     ++repetition) {
                    nbody::CudaTiming timing{};
                    const nbody::Accelerations accelerations =
                        nbody::compute_acceleration_cuda_tiled(
                            bodies,
                            options.epsilon,
                            options.block_size,
                            &timing);
                    if (accelerations.size() != count ||
                        !std::isfinite(accelerations.front().x)) {
                        throw std::runtime_error("force benchmark produced invalid output");
                    }
                    output << "cuda_tiled_force," << nbody::to_string(options.family)
                           << ',' << count << ',' << seed << ',' << repetition << ','
                           << timing.allocation_ms << ',' << timing.host_to_device_ms
                           << ',' << timing.kernel_ms << ',' << timing.device_to_host_ms
                           << ',' << timing.cleanup_ms << ',' << timing.total_ms << '\n';
                }
                std::cout << "benchmarked tiled CUDA force at N=" << count
                          << ", seed=" << seed << '\n';
            }
        }
        std::cout << "force benchmark written to: " << options.output << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
