#include "nbody/simulation.hpp"

#include <algorithm>
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
    std::vector<std::size_t> body_counts{128, 256};
    std::size_t trajectories_per_count{4};
    std::size_t snapshots{4};
    std::size_t stride{10};
    double dt{0.001};
    double epsilon{0.01};
    std::uint64_t base_seed{12345};
    int block_size{64};
    std::size_t verify_every{1};
    double verification_tolerance{1e-10};
    std::filesystem::path output{"data/raw/pilot_cuda.csv"};
    bool overwrite{false};
};

const char* require_value(int& index, int argc, char** argv) {
    if (++index >= argc) {
        throw std::invalid_argument(
            "missing value for " + std::string(argv[index - 1])
        );
    }
    return argv[index];
}

std::size_t parse_positive_size(const std::string& text, const char* option) {
    std::size_t consumed = 0;
    const long long value = std::stoll(text, &consumed);
    if (consumed != text.size() || value <= 0) {
        throw std::invalid_argument(std::string(option) + " must be positive");
    }
    return static_cast<std::size_t>(value);
}

std::vector<std::size_t> parse_body_counts(const std::string& text) {
    std::vector<std::size_t> counts;
    std::stringstream stream(text);
    std::string token;
    while (std::getline(stream, token, ',')) {
        counts.push_back(parse_positive_size(token, "--body-counts"));
    }
    if (counts.empty()) {
        throw std::invalid_argument("--body-counts cannot be empty");
    }
    return counts;
}

void print_help(const char* program_name) {
    std::cout
        << "Usage: " << program_name << " [options]\n\n"
        << "  --body-counts LIST            Comma-separated counts (default: 128,256)\n"
        << "  --trajectories-per-count N    Independent seeds per count (default: 4)\n"
        << "  --snapshots N                 Snapshots per trajectory (default: 4)\n"
        << "  --stride N                    Steps between snapshots (default: 10)\n"
        << "  --dt VALUE                    Integration time step (default: 0.001)\n"
        << "  --epsilon VALUE               Force softening (default: 0.01)\n"
        << "  --base-seed N                 First trajectory seed (default: 12345)\n"
        << "  --block-size N                CUDA block size (default: 64)\n"
        << "  --verify-every N              Verify every Nth snapshot; 0 disables\n"
        << "  --verification-tolerance V    Naive/tiled tolerance (default: 1e-10)\n"
        << "  --output FILE                 Output CSV path\n"
        << "  --overwrite                   Replace an existing output file\n";
}

Options parse_arguments(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view option{argv[i]};
        if (option == "--help" || option == "-h") {
            print_help(argv[0]);
            std::exit(0);
        } else if (option == "--body-counts") {
            options.body_counts = parse_body_counts(require_value(i, argc, argv));
        } else if (option == "--trajectories-per-count") {
            options.trajectories_per_count = parse_positive_size(
                require_value(i, argc, argv),
                "--trajectories-per-count"
            );
        } else if (option == "--snapshots") {
            options.snapshots = parse_positive_size(
                require_value(i, argc, argv),
                "--snapshots"
            );
        } else if (option == "--stride") {
            options.stride = parse_positive_size(
                require_value(i, argc, argv),
                "--stride"
            );
        } else if (option == "--dt") {
            options.dt = std::stod(require_value(i, argc, argv));
        } else if (option == "--epsilon") {
            options.epsilon = std::stod(require_value(i, argc, argv));
        } else if (option == "--base-seed") {
            const std::string value = require_value(i, argc, argv);
            if (!value.empty() && value.front() == '-') {
                throw std::invalid_argument("--base-seed cannot be negative");
            }
            options.base_seed = std::stoull(value);
        } else if (option == "--block-size") {
            options.block_size = static_cast<int>(parse_positive_size(
                require_value(i, argc, argv),
                "--block-size"
            ));
        } else if (option == "--verify-every") {
            const std::string value = require_value(i, argc, argv);
            if (!value.empty() && value.front() == '-') {
                throw std::invalid_argument("--verify-every cannot be negative");
            }
            options.verify_every = std::stoull(value);
        } else if (option == "--verification-tolerance") {
            options.verification_tolerance = std::stod(
                require_value(i, argc, argv)
            );
        } else if (option == "--output") {
            options.output = require_value(i, argc, argv);
        } else if (option == "--overwrite") {
            options.overwrite = true;
        } else {
            throw std::invalid_argument("unknown option: " + std::string(option));
        }
    }

    if (options.dt <= 0.0) {
        throw std::invalid_argument("--dt must be positive");
    }
    if (options.epsilon < 0.0) {
        throw std::invalid_argument("--epsilon cannot be negative");
    }
    if (options.verification_tolerance < 0.0) {
        throw std::invalid_argument("--verification-tolerance cannot be negative");
    }
    if (options.output.empty()) {
        throw std::invalid_argument("--output cannot be empty");
    }
    return options;
}

double verify_accelerations(
    const nbody::Accelerations& tiled,
    const nbody::Accelerations& naive,
    double tolerance
) {
    if (tiled.size() != naive.size()) {
        throw std::runtime_error("naive and tiled label counts differ");
    }

    double maximum_absolute_error = 0.0;
    for (std::size_t i = 0; i < tiled.size(); ++i) {
        const double differences[] = {
            std::abs(tiled[i].x - naive[i].x),
            std::abs(tiled[i].y - naive[i].y),
            std::abs(tiled[i].z - naive[i].z),
        };
        const double scales[] = {
            1.0 + std::abs(naive[i].x),
            1.0 + std::abs(naive[i].y),
            1.0 + std::abs(naive[i].z),
        };
        for (int component = 0; component < 3; ++component) {
            maximum_absolute_error = std::max(
                maximum_absolute_error,
                differences[component]
            );
            if (differences[component] > tolerance * scales[component]) {
                throw std::runtime_error(
                    "naive/tiled acceleration verification failed at body " +
                    std::to_string(i)
                );
            }
        }
    }
    return maximum_absolute_error;
}

void write_snapshot(
    std::ofstream& output,
    std::size_t trajectory_id,
    std::size_t snapshot_id,
    std::size_t time_step,
    double dt,
    std::uint64_t seed,
    double epsilon,
    const nbody::Bodies& bodies,
    const nbody::Accelerations& accelerations
) {
    const double physical_time = static_cast<double>(time_step) * dt;
    const double characteristic_radius = std::sqrt(3.0);
    double total_mass = 0.0;
    for (const nbody::Body& body : bodies) {
        total_mass += body.mass;
    }

    for (std::size_t body_id = 0; body_id < bodies.size(); ++body_id) {
        const nbody::Body& body = bodies[body_id];
        const nbody::Acceleration& acceleration = accelerations[body_id];
        output
            << trajectory_id << ','
            << snapshot_id << ','
            << time_step << ','
            << physical_time << ','
            << dt << ','
            << body_id << ','
            << bodies.size() << ','
            << seed << ','
            << "uniform_cube" << ','
            << characteristic_radius << ','
            << total_mass << ','
            << epsilon << ','
            << body.x << ',' << body.y << ',' << body.z << ','
            << body.vx << ',' << body.vy << ',' << body.vz << ','
            << body.mass << ','
            << acceleration.x << ','
            << acceleration.y << ','
            << acceleration.z << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_arguments(argc, argv);
        if (std::filesystem::exists(options.output) && !options.overwrite) {
            throw std::runtime_error(
                "output already exists; use --overwrite: " +
                options.output.string()
            );
        }
        if (options.output.has_parent_path()) {
            std::filesystem::create_directories(options.output.parent_path());
        }

        std::ofstream output(options.output, std::ios::trunc);
        if (!output) {
            throw std::runtime_error(
                "could not open output file: " + options.output.string()
            );
        }
        output << std::setprecision(17);
        output
            << "trajectory_id,snapshot_id,time_step,time,dt,body_id,"
            << "particle_count,seed,initial_condition,characteristic_radius,"
            << "total_mass,epsilon,x,y,z,vx,vy,vz,mass,ax,ay,az\n";

        std::size_t trajectory_id = 0;
        std::size_t global_snapshot_id = 0;
        double maximum_verification_error = 0.0;

        for (const std::size_t body_count : options.body_counts) {
            for (std::size_t trajectory = 0;
                 trajectory < options.trajectories_per_count;
                 ++trajectory) {
                const std::uint64_t seed = options.base_seed + trajectory_id;
                nbody::Bodies bodies = nbody::initialize_bodies(body_count, seed);

                std::cout
                    << "trajectory " << trajectory_id
                    << ": bodies=" << body_count
                    << ", seed=" << seed << '\n';

                for (std::size_t snapshot = 0;
                     snapshot < options.snapshots;
                     ++snapshot) {
                    const std::size_t time_step = snapshot * options.stride;
                    const nbody::Accelerations tiled =
                        nbody::compute_acceleration_cuda_tiled(
                            bodies,
                            options.epsilon,
                            options.block_size
                        );

                    if (options.verify_every > 0 &&
                        global_snapshot_id % options.verify_every == 0) {
                        const nbody::Accelerations naive =
                            nbody::compute_acceleration_cuda_naive(
                                bodies,
                                options.epsilon,
                                options.block_size
                            );
                        maximum_verification_error = std::max(
                            maximum_verification_error,
                            verify_accelerations(
                                tiled,
                                naive,
                                options.verification_tolerance
                            )
                        );
                    }

                    write_snapshot(
                        output,
                        trajectory_id,
                        snapshot,
                        time_step,
                        options.dt,
                        seed,
                        options.epsilon,
                        bodies,
                        tiled
                    );
                    if (!output) {
                        throw std::runtime_error("failed while writing dataset");
                    }

                    ++global_snapshot_id;
                    if (snapshot + 1 < options.snapshots) {
                        bodies = nbody::simulate_cuda_tiled(
                            bodies,
                            options.stride,
                            options.dt,
                            options.epsilon,
                            options.block_size
                        );
                    }
                }
                ++trajectory_id;
            }
        }

        std::cout
            << "pilot dataset written to: " << options.output.string() << '\n'
            << "trajectories: " << trajectory_id << '\n'
            << "snapshots: " << global_snapshot_id << '\n'
            << "maximum naive/tiled acceleration difference: "
            << maximum_verification_error << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
