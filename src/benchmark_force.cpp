#include "nbody/initial_conditions.hpp"
#include "nbody/simulation.hpp"

#include <algorithm>
#include <chrono>
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

enum class ForceImplementation { serial, openmp, cuda_naive, cuda_tiled };

struct Options {
    std::vector<std::size_t> body_counts{4096, 8192, 16384};
    std::uint64_t base_seed{600000};
    std::size_t trajectories_per_count{1};
    std::size_t repetitions{5};
    double epsilon{0.01};
    int threads{0};
    int block_size{64};
    std::vector<ForceImplementation> implementations{
#ifdef NBODY_HAS_CUDA
        ForceImplementation::cuda_tiled
#else
        ForceImplementation::serial
#endif
    };
    nbody::InitialConditionFamily family{nbody::InitialConditionFamily::uniform_cube};
    std::filesystem::path output{"results/cuda_force_large_n.csv"};
    bool append{};
};

std::string_view implementation_name(ForceImplementation implementation) {
    switch (implementation) {
        case ForceImplementation::serial: return "serial_force";
        case ForceImplementation::openmp: return "openmp_force";
        case ForceImplementation::cuda_naive: return "cuda_naive_force";
        case ForceImplementation::cuda_tiled: return "cuda_tiled_force";
    }
    return "unknown_force";
}

std::vector<ForceImplementation> implementations(const std::string& text) {
    std::vector<ForceImplementation> values;
    std::stringstream stream(text);
    std::string token;
    while (std::getline(stream, token, ',')) {
        if (token == "serial") {
            values.push_back(ForceImplementation::serial);
        } else if (token == "openmp") {
            values.push_back(ForceImplementation::openmp);
        } else if (token == "cuda_naive") {
            values.push_back(ForceImplementation::cuda_naive);
        } else if (token == "cuda_tiled") {
            values.push_back(ForceImplementation::cuda_tiled);
        } else {
            throw std::invalid_argument("unknown force implementation: " + token);
        }
    }
    if (values.empty()) {
        throw std::invalid_argument("--implementations cannot be empty");
    }
    return values;
}

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
        } else if (option == "--threads") {
            options.threads = std::stoi(require_value(index, argc, argv));
        } else if (option == "--implementations") {
            options.implementations = implementations(
                require_value(index, argc, argv));
        } else if (option == "--block-size") {
            options.block_size = static_cast<int>(positive_size(
                require_value(index, argc, argv), "--block-size"));
        } else if (option == "--initial-condition") {
            options.family = nbody::parse_initial_condition(
                require_value(index, argc, argv));
        } else if (option == "--output") {
            options.output = require_value(index, argc, argv);
        } else if (option == "--append") {
            options.append = true;
        } else if (option == "--help" || option == "-h") {
            std::cout
                << "Usage: " << argv[0] << " [options]\n"
                << "  --body-counts LIST\n"
                << "  --base-seed N\n"
                << "  --repetitions N\n"
                << "  --trajectories-per-count N\n"
                << "  --epsilon VALUE\n"
                << "  --implementations serial,openmp,cuda_naive,cuda_tiled\n"
                << "  --threads N\n"
                << "  --block-size N\n"
                << "  --initial-condition NAME\n"
                << "  --output FILE\n"
                << "  --append\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown option: " + std::string(option));
        }
    }
    if (!std::isfinite(options.epsilon) || options.epsilon < 0.0 ||
        options.threads < 0 || options.block_size <= 0 || options.output.empty()) {
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
        const std::string output_header =
            "implementation,initial_condition,bodies,seed,repetition,threads,"
            "allocation_ms,h2d_ms,kernel_ms,d2h_ms,cleanup_ms,"
            "cuda_reported_total_ms,total_ms";
        bool write_header = true;
        if (options.append && std::filesystem::exists(options.output)) {
            std::ifstream existing(options.output);
            std::string existing_header;
            std::getline(existing, existing_header);
            if (!existing_header.empty() && existing_header != output_header) {
                throw std::runtime_error(
                    "force benchmark output schema does not match: " +
                    options.output.string());
            }
            write_header = existing_header.empty();
        }
        const auto mode = options.append ? std::ios::app : std::ios::trunc;
        std::ofstream output(options.output, mode);
        if (!output) {
            throw std::runtime_error("could not open output: " + options.output.string());
        }
        output << std::setprecision(17);
        if (write_header) {
            output << output_header << '\n';
        }

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
                for (const ForceImplementation implementation :
                     options.implementations) {
                    int threads_used = 0;
                    if (implementation == ForceImplementation::serial) {
                        threads_used = 1;
                    } else if (implementation == ForceImplementation::openmp) {
                        threads_used = nbody::openmp_thread_count(options.threads);
                    }

                    const auto compute = [&](nbody::CudaTiming* timing) {
                        switch (implementation) {
                            case ForceImplementation::serial:
                                return nbody::compute_acceleration_serial(
                                    bodies, options.epsilon);
                            case ForceImplementation::openmp:
                                return nbody::compute_acceleration_openmp(
                                    bodies, options.epsilon, threads_used);
                            case ForceImplementation::cuda_naive:
#ifdef NBODY_HAS_CUDA
                                (void)timing;
                                return nbody::compute_acceleration_cuda_naive(
                                    bodies, options.epsilon, options.block_size);
#else
                                (void)timing;
                                throw std::runtime_error(
                                    "CUDA support is not enabled in this build");
#endif
                            case ForceImplementation::cuda_tiled:
#ifdef NBODY_HAS_CUDA
                                return nbody::compute_acceleration_cuda_tiled(
                                    bodies,
                                    options.epsilon,
                                    options.block_size,
                                    timing);
#else
                                (void)timing;
                                throw std::runtime_error(
                                    "CUDA support is not enabled in this build");
#endif
                        }
                        throw std::runtime_error("invalid force implementation");
                    };

                    (void)compute(nullptr);
                    for (std::size_t repetition = 1;
                         repetition <= options.repetitions;
                         ++repetition) {
                        nbody::CudaTiming timing{};
                        const auto started = std::chrono::steady_clock::now();
                        const nbody::Accelerations accelerations = compute(&timing);
                        const auto stopped = std::chrono::steady_clock::now();
                        const double total_ms =
                            std::chrono::duration<double, std::milli>(
                                stopped - started).count();
                        const bool finite =
                            accelerations.size() == count &&
                            std::all_of(
                                accelerations.begin(),
                                accelerations.end(),
                                [](const nbody::Acceleration& acceleration) {
                                    return std::isfinite(acceleration.x) &&
                                           std::isfinite(acceleration.y) &&
                                           std::isfinite(acceleration.z);
                                });
                        if (!finite) {
                            throw std::runtime_error(
                                "force benchmark produced invalid output");
                        }
                        output << implementation_name(implementation) << ','
                               << nbody::to_string(options.family) << ',' << count
                               << ',' << seed << ',' << repetition << ','
                               << threads_used << ',' << timing.allocation_ms << ','
                               << timing.host_to_device_ms << ',' << timing.kernel_ms
                               << ',' << timing.device_to_host_ms << ','
                               << timing.cleanup_ms << ',' << timing.total_ms << ','
                               << total_ms << '\n';
                    }
                    std::cout << "benchmarked "
                              << implementation_name(implementation)
                              << " at N=" << count << ", seed=" << seed
                              << ", threads=" << threads_used << '\n';
                }
            }
        }
        std::cout << "force benchmark written to: " << options.output << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
