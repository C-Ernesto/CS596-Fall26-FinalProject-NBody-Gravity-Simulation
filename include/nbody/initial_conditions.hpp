#pragma once

#include "nbody/simulation.hpp"

#include <cstdint>
#include <string>
#include <string_view>

namespace nbody {

enum class InitialConditionFamily {
    uniform_cube,
    plummer_sphere,
    two_cluster_collision,
    rotating_disk,
};

struct InitialConditionParameters {
    double position_scale{1.0};
    double velocity_scale{0.01};
    double cluster_separation{4.0};
    double cluster_speed{0.2};
    double disk_thickness{0.05};
};

InitialConditionFamily parse_initial_condition(std::string_view name);
std::string to_string(InitialConditionFamily family);

Bodies initialize_bodies_for_family(
    std::size_t count,
    std::uint64_t seed,
    InitialConditionFamily family,
    const InitialConditionParameters& parameters
);

double characteristic_radius(
    InitialConditionFamily family,
    const InitialConditionParameters& parameters
);

}  // namespace nbody
