#include "nbody/initial_conditions.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

namespace nbody {
namespace {

constexpr double pi = 3.14159265358979323846;

struct Point3 {
    double x{};
    double y{};
    double z{};
};

Point3 isotropic_direction(std::mt19937_64& generator) {
    std::normal_distribution<double> normal(0.0, 1.0);
    Point3 direction{normal(generator), normal(generator), normal(generator)};
    const double length = std::sqrt(
        direction.x * direction.x + direction.y * direction.y +
        direction.z * direction.z
    );
    direction.x /= length;
    direction.y /= length;
    direction.z /= length;
    return direction;
}

Point3 plummer_position(
    std::mt19937_64& generator,
    double scale
) {
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    double radius = 0.0;
    do {
        const double u = std::max(uniform(generator), 1e-12);
        radius = scale / std::sqrt(std::pow(u, -2.0 / 3.0) - 1.0);
    } while (radius > 10.0 * scale);

    const Point3 direction = isotropic_direction(generator);
    return {
        radius * direction.x,
        radius * direction.y,
        radius * direction.z,
    };
}

void remove_center_of_mass_motion(Bodies& bodies) {
    double total_mass = 0.0;
    double center_x = 0.0, center_y = 0.0, center_z = 0.0;
    double mean_vx = 0.0, mean_vy = 0.0, mean_vz = 0.0;
    for (const Body& body : bodies) {
        total_mass += body.mass;
        center_x += body.mass * body.x;
        center_y += body.mass * body.y;
        center_z += body.mass * body.z;
        mean_vx += body.mass * body.vx;
        mean_vy += body.mass * body.vy;
        mean_vz += body.mass * body.vz;
    }
    center_x /= total_mass;
    center_y /= total_mass;
    center_z /= total_mass;
    mean_vx /= total_mass;
    mean_vy /= total_mass;
    mean_vz /= total_mass;

    for (Body& body : bodies) {
        body.x -= center_x;
        body.y -= center_y;
        body.z -= center_z;
        body.vx -= mean_vx;
        body.vy -= mean_vy;
        body.vz -= mean_vz;
    }
}

void validate_parameters(const InitialConditionParameters& parameters) {
    if (parameters.position_scale <= 0.0) {
        throw std::invalid_argument("position scale must be positive");
    }
    if (parameters.velocity_scale < 0.0) {
        throw std::invalid_argument("velocity scale cannot be negative");
    }
    if (parameters.cluster_separation <= 0.0) {
        throw std::invalid_argument("cluster separation must be positive");
    }
    if (parameters.cluster_speed < 0.0) {
        throw std::invalid_argument("cluster speed cannot be negative");
    }
    if (parameters.disk_thickness < 0.0) {
        throw std::invalid_argument("disk thickness cannot be negative");
    }
}

}  // namespace

InitialConditionFamily parse_initial_condition(std::string_view name) {
    if (name == "uniform_cube") return InitialConditionFamily::uniform_cube;
    if (name == "plummer_sphere") return InitialConditionFamily::plummer_sphere;
    if (name == "two_cluster_collision") {
        return InitialConditionFamily::two_cluster_collision;
    }
    if (name == "rotating_disk") return InitialConditionFamily::rotating_disk;
    throw std::invalid_argument("unknown initial-condition family: " + std::string(name));
}

std::string to_string(InitialConditionFamily family) {
    switch (family) {
        case InitialConditionFamily::uniform_cube: return "uniform_cube";
        case InitialConditionFamily::plummer_sphere: return "plummer_sphere";
        case InitialConditionFamily::two_cluster_collision:
            return "two_cluster_collision";
        case InitialConditionFamily::rotating_disk: return "rotating_disk";
    }
    return "unknown";
}

Bodies initialize_bodies_for_family(
    std::size_t count,
    std::uint64_t seed,
    InitialConditionFamily family,
    const InitialConditionParameters& parameters
) {
    if (count == 0) {
        throw std::invalid_argument("body count must be positive");
    }
    validate_parameters(parameters);

    std::mt19937_64 generator(seed);
    std::uniform_real_distribution<double> uniform(-1.0, 1.0);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::normal_distribution<double> velocity_noise(
        0.0,
        parameters.velocity_scale
    );
    std::normal_distribution<double> disk_height(
        0.0,
        parameters.disk_thickness
    );

    Bodies bodies(count);
    const double mass = 1.0 / static_cast<double>(count);
    for (std::size_t i = 0; i < count; ++i) {
        Body& body = bodies[i];
        body.mass = mass;

        switch (family) {
            case InitialConditionFamily::uniform_cube:
                body.x = parameters.position_scale * uniform(generator);
                body.y = parameters.position_scale * uniform(generator);
                body.z = parameters.position_scale * uniform(generator);
                body.vx = parameters.velocity_scale * uniform(generator);
                body.vy = parameters.velocity_scale * uniform(generator);
                body.vz = parameters.velocity_scale * uniform(generator);
                break;

            case InitialConditionFamily::plummer_sphere: {
                const Point3 position = plummer_position(
                    generator,
                    parameters.position_scale
                );
                body.x = position.x;
                body.y = position.y;
                body.z = position.z;
                body.vx = velocity_noise(generator);
                body.vy = velocity_noise(generator);
                body.vz = velocity_noise(generator);
                break;
            }

            case InitialConditionFamily::two_cluster_collision: {
                const bool left_cluster = i < (count + 1) / 2;
                const Point3 local_position = plummer_position(
                    generator,
                    parameters.position_scale
                );
                const double center = 0.5 * parameters.cluster_separation;
                body.x = local_position.x + (left_cluster ? -center : center);
                body.y = local_position.y;
                body.z = local_position.z;
                body.vx = (left_cluster ? parameters.cluster_speed : -parameters.cluster_speed) +
                          velocity_noise(generator);
                body.vy = velocity_noise(generator);
                body.vz = velocity_noise(generator);
                break;
            }

            case InitialConditionFamily::rotating_disk: {
                const double radius = parameters.position_scale * std::sqrt(unit(generator));
                const double angle = 2.0 * pi * unit(generator);
                body.x = radius * std::cos(angle);
                body.y = radius * std::sin(angle);
                body.z = disk_height(generator);

                const double enclosed_mass = std::max(
                    radius * radius /
                        (parameters.position_scale * parameters.position_scale),
                    1.0 / static_cast<double>(count)
                );
                const double circular_speed = std::sqrt(
                    enclosed_mass / std::max(radius, 1e-6)
                );
                body.vx = -circular_speed * std::sin(angle) +
                          velocity_noise(generator);
                body.vy = circular_speed * std::cos(angle) +
                          velocity_noise(generator);
                body.vz = velocity_noise(generator);
                break;
            }
        }
    }

    remove_center_of_mass_motion(bodies);
    return bodies;
}

double characteristic_radius(
    InitialConditionFamily family,
    const InitialConditionParameters& parameters
) {
    validate_parameters(parameters);
    switch (family) {
        case InitialConditionFamily::uniform_cube:
            return std::sqrt(3.0) * parameters.position_scale;
        case InitialConditionFamily::plummer_sphere:
            return parameters.position_scale;
        case InitialConditionFamily::two_cluster_collision:
            return 0.5 * parameters.cluster_separation + parameters.position_scale;
        case InitialConditionFamily::rotating_disk:
            return parameters.position_scale;
    }
    throw std::invalid_argument("unknown initial-condition family");
}

}  // namespace nbody
