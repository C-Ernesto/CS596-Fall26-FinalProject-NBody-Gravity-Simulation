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
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace {

struct Options {
    std::filesystem::path input;
    std::filesystem::path output{"results/cpu_label_validation.csv"};
    std::size_t trajectories_per_count{1};
    std::set<std::size_t> snapshot_ids{0, 7};
    double tolerance{1.0e-10};
};

struct Snapshot {
    bool selected{};
    std::size_t trajectory_id{};
    std::size_t snapshot_id{};
    std::size_t time_step{};
    std::size_t particle_count{};
    std::uint64_t seed{};
    std::string family;
    double epsilon{};
    nbody::Bodies bodies;
    nbody::Accelerations labels;
};

const char* require_value(int& index, int argc, char** argv) {
    if (++index >= argc) {
        throw std::invalid_argument("missing value for " + std::string(argv[index - 1]));
    }
    return argv[index];
}

std::vector<std::string> split(const std::string& text) {
    std::vector<std::string> values;
    std::stringstream stream(text);
    std::string value;
    while (std::getline(stream, value, ',')) {
        values.push_back(value);
    }
    return values;
}

std::set<std::size_t> snapshot_ids(const std::string& text) {
    std::set<std::size_t> values;
    for (const std::string& token : split(text)) {
        std::size_t consumed = 0;
        const unsigned long long value = std::stoull(token, &consumed);
        if (consumed != token.size()) {
            throw std::invalid_argument("invalid --snapshot-ids value: " + token);
        }
        values.insert(static_cast<std::size_t>(value));
    }
    if (values.empty()) {
        throw std::invalid_argument("--snapshot-ids cannot be empty");
    }
    return values;
}

Options parse_arguments(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view option{argv[index]};
        if (option == "--input") {
            options.input = require_value(index, argc, argv);
        } else if (option == "--output") {
            options.output = require_value(index, argc, argv);
        } else if (option == "--trajectories-per-count") {
            options.trajectories_per_count = static_cast<std::size_t>(
                std::stoull(require_value(index, argc, argv)));
        } else if (option == "--snapshot-ids") {
            options.snapshot_ids = snapshot_ids(require_value(index, argc, argv));
        } else if (option == "--tolerance") {
            options.tolerance = std::stod(require_value(index, argc, argv));
        } else if (option == "--help" || option == "-h") {
            std::cout
                << "Usage: " << argv[0] << " --input FILE [options]\n"
                << "  --output FILE\n"
                << "  --trajectories-per-count N\n"
                << "  --snapshot-ids LIST\n"
                << "  --tolerance VALUE\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown option: " + std::string(option));
        }
    }
    if (options.input.empty() || !std::filesystem::is_regular_file(options.input)) {
        throw std::invalid_argument("--input must name an existing CSV file");
    }
    if (options.output.empty() || options.trajectories_per_count == 0 ||
        !std::isfinite(options.tolerance) || options.tolerance < 0.0) {
        throw std::invalid_argument("invalid dataset validation configuration");
    }
    return options;
}

std::size_t column(
    const std::unordered_map<std::string, std::size_t>& columns,
    const std::string& name
) {
    const auto found = columns.find(name);
    if (found == columns.end()) {
        throw std::runtime_error("dataset CSV is missing column: " + name);
    }
    return found->second;
}

void prepare_output(const std::filesystem::path& path, const std::string& header) {
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }
    if (std::filesystem::exists(path)) {
        std::ifstream existing(path);
        std::string existing_header;
        std::getline(existing, existing_header);
        if (!existing_header.empty() && existing_header != header) {
            throw std::runtime_error(
                "validation output schema does not match: " + path.string());
        }
        return;
    }
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("could not create output: " + path.string());
    }
    output << header << '\n';
}

bool validate_snapshot(
    const Snapshot& snapshot,
    const Options& options,
    std::ofstream& output
) {
    if (!snapshot.selected) {
        return true;
    }
    if (snapshot.bodies.size() != snapshot.particle_count ||
        snapshot.labels.size() != snapshot.particle_count) {
        throw std::runtime_error("selected snapshot has an incomplete body count");
    }
    const nbody::Accelerations serial =
        nbody::compute_acceleration_serial(snapshot.bodies, snapshot.epsilon);
    double maximum_absolute_error = 0.0;
    double maximum_scaled_error = 0.0;
    double squared_error = 0.0;
    double squared_reference = 0.0;
    for (std::size_t index = 0; index < serial.size(); ++index) {
        const double actual[] = {
            snapshot.labels[index].x,
            snapshot.labels[index].y,
            snapshot.labels[index].z,
        };
        const double reference[] = {
            serial[index].x,
            serial[index].y,
            serial[index].z,
        };
        for (int component = 0; component < 3; ++component) {
            const double difference = std::abs(actual[component] - reference[component]);
            maximum_absolute_error = std::max(maximum_absolute_error, difference);
            maximum_scaled_error = std::max(
                maximum_scaled_error,
                difference / (1.0 + std::abs(reference[component])));
            squared_error += difference * difference;
            squared_reference += reference[component] * reference[component];
        }
    }
    const double relative_l2 =
        std::sqrt(squared_error / std::max(squared_reference, 1.0e-300));
    const bool passed = maximum_scaled_error <= options.tolerance;
    output << std::setprecision(17) << options.input.filename().string() << ','
           << snapshot.family << ',' << snapshot.particle_count << ','
           << snapshot.seed << ',' << snapshot.trajectory_id << ','
           << snapshot.snapshot_id << ',' << snapshot.time_step << ','
           << maximum_absolute_error << ',' << maximum_scaled_error << ','
           << relative_l2 << ',' << (passed ? "true" : "false") << '\n';
    std::cout << (passed ? "passed" : "FAILED") << ": "
              << snapshot.family << " N=" << snapshot.particle_count
              << " seed=" << snapshot.seed
              << " snapshot=" << snapshot.snapshot_id
              << " max_scaled_error=" << maximum_scaled_error << '\n';
    return passed;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_arguments(argc, argv);
        const std::string output_header =
            "input_file,initial_condition,bodies,seed,trajectory_id,snapshot_id,"
            "time_step,max_absolute_error,max_scaled_error,relative_l2,passed";
        prepare_output(options.output, output_header);
        std::ofstream output(options.output, std::ios::app);
        if (!output) {
            throw std::runtime_error("could not append output: " + options.output.string());
        }

        std::ifstream input(options.input);
        std::string line;
        if (!std::getline(input, line)) {
            throw std::runtime_error("dataset CSV is empty");
        }
        const std::vector<std::string> header = split(line);
        std::unordered_map<std::string, std::size_t> columns;
        for (std::size_t index = 0; index < header.size(); ++index) {
            columns.emplace(header[index], index);
        }
        const std::size_t trajectory_column = column(columns, "trajectory_id");
        const std::size_t snapshot_column = column(columns, "snapshot_id");
        const std::size_t time_step_column = column(columns, "time_step");
        const std::size_t body_column = column(columns, "body_id");
        const std::size_t count_column = column(columns, "particle_count");
        const std::size_t seed_column = column(columns, "seed");
        const std::size_t family_column = column(columns, "initial_condition");
        const std::size_t epsilon_column = column(columns, "epsilon");
        const std::size_t x_column = column(columns, "x");
        const std::size_t y_column = column(columns, "y");
        const std::size_t z_column = column(columns, "z");
        const std::size_t vx_column = column(columns, "vx");
        const std::size_t vy_column = column(columns, "vy");
        const std::size_t vz_column = column(columns, "vz");
        const std::size_t mass_column = column(columns, "mass");
        const std::size_t ax_column = column(columns, "ax");
        const std::size_t ay_column = column(columns, "ay");
        const std::size_t az_column = column(columns, "az");

        std::map<std::size_t, std::set<std::size_t>> selected_trajectories;
        Snapshot snapshot;
        bool have_snapshot = false;
        bool all_passed = true;
        std::size_t validated = 0;
        const auto finish_snapshot = [&]() {
            if (have_snapshot && snapshot.selected) {
                all_passed = validate_snapshot(snapshot, options, output) && all_passed;
                ++validated;
            }
        };

        while (std::getline(input, line)) {
            const std::vector<std::string> values = split(line);
            if (values.size() != header.size()) {
                throw std::runtime_error("malformed dataset CSV row");
            }
            const std::size_t trajectory = std::stoull(values[trajectory_column]);
            const std::size_t snapshot_id = std::stoull(values[snapshot_column]);
            if (!have_snapshot || trajectory != snapshot.trajectory_id ||
                snapshot_id != snapshot.snapshot_id) {
                finish_snapshot();
                snapshot = Snapshot{};
                have_snapshot = true;
                snapshot.trajectory_id = trajectory;
                snapshot.snapshot_id = snapshot_id;
                snapshot.time_step = std::stoull(values[time_step_column]);
                snapshot.particle_count = std::stoull(values[count_column]);
                snapshot.seed = std::stoull(values[seed_column]);
                snapshot.family = values[family_column];
                snapshot.epsilon = std::stod(values[epsilon_column]);
                auto& selected = selected_trajectories[snapshot.particle_count];
                if (selected.count(trajectory) == 0 &&
                    selected.size() < options.trajectories_per_count) {
                    selected.insert(trajectory);
                }
                snapshot.selected =
                    selected.count(trajectory) != 0 &&
                    options.snapshot_ids.count(snapshot_id) != 0;
                if (snapshot.selected) {
                    snapshot.bodies.reserve(snapshot.particle_count);
                    snapshot.labels.reserve(snapshot.particle_count);
                }
            }
            if (snapshot.selected) {
                const std::size_t body_id = std::stoull(values[body_column]);
                if (body_id != snapshot.bodies.size()) {
                    throw std::runtime_error("body IDs are not contiguous within snapshot");
                }
                snapshot.bodies.push_back(nbody::Body{
                    std::stod(values[x_column]),
                    std::stod(values[y_column]),
                    std::stod(values[z_column]),
                    std::stod(values[vx_column]),
                    std::stod(values[vy_column]),
                    std::stod(values[vz_column]),
                    std::stod(values[mass_column]),
                });
                snapshot.labels.push_back(nbody::Acceleration{
                    std::stod(values[ax_column]),
                    std::stod(values[ay_column]),
                    std::stod(values[az_column]),
                });
            }
        }
        finish_snapshot();
        if (validated == 0) {
            throw std::runtime_error("no snapshots matched the validation selection");
        }
        if (!all_passed) {
            throw std::runtime_error("one or more dataset snapshots failed serial validation");
        }
        std::cout << "validated " << validated << " snapshots from "
                  << options.input << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
