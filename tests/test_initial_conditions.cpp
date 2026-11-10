#include "nbody/initial_conditions.hpp"

#include <cmath>
#include <iostream>

int main() {
    const nbody::InitialConditionFamily families[] = {
        nbody::InitialConditionFamily::uniform_cube,
        nbody::InitialConditionFamily::plummer_sphere,
        nbody::InitialConditionFamily::two_cluster_collision,
        nbody::InitialConditionFamily::rotating_disk,
    };
    const nbody::InitialConditionParameters parameters{};

    for (const nbody::InitialConditionFamily family : families) {
        const nbody::Bodies first = nbody::initialize_bodies_for_family(
            257,
            12345,
            family,
            parameters
        );
        const nbody::Bodies second = nbody::initialize_bodies_for_family(
            257,
            12345,
            family,
            parameters
        );

        if (first.size() != 257 || first.size() != second.size()) {
            std::cerr << "Unexpected body count for " << nbody::to_string(family) << '\n';
            return 1;
        }

        double total_mass = 0.0;
        double center_x = 0.0, center_y = 0.0, center_z = 0.0;
        double momentum_x = 0.0, momentum_y = 0.0, momentum_z = 0.0;
        for (std::size_t i = 0; i < first.size(); ++i) {
            const nbody::Body& body = first[i];
            const nbody::Body& repeated = second[i];
            const bool finite =
                std::isfinite(body.x) && std::isfinite(body.y) &&
                std::isfinite(body.z) && std::isfinite(body.vx) &&
                std::isfinite(body.vy) && std::isfinite(body.vz) &&
                std::isfinite(body.mass);
            const bool deterministic =
                body.x == repeated.x && body.y == repeated.y &&
                body.z == repeated.z && body.vx == repeated.vx &&
                body.vy == repeated.vy && body.vz == repeated.vz &&
                body.mass == repeated.mass;
            if (!finite || !deterministic || body.mass <= 0.0) {
                std::cerr << "Invalid body for " << nbody::to_string(family)
                          << " at index " << i << '\n';
                return 1;
            }

            total_mass += body.mass;
            center_x += body.mass * body.x;
            center_y += body.mass * body.y;
            center_z += body.mass * body.z;
            momentum_x += body.mass * body.vx;
            momentum_y += body.mass * body.vy;
            momentum_z += body.mass * body.vz;
        }

        const double tolerance = 1e-12;
        const bool conserved =
            std::abs(total_mass - 1.0) < tolerance &&
            std::abs(center_x) < tolerance &&
            std::abs(center_y) < tolerance &&
            std::abs(center_z) < tolerance &&
            std::abs(momentum_x) < tolerance &&
            std::abs(momentum_y) < tolerance &&
            std::abs(momentum_z) < tolerance;
        if (!conserved || nbody::characteristic_radius(family, parameters) <= 0.0) {
            std::cerr << "Invalid aggregate state for "
                      << nbody::to_string(family) << '\n';
            return 1;
        }
    }

    return 0;
}
