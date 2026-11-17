#pragma once

// Internal helpers shared by the serial and OpenMP CPU backends.
//
// Both backends call the same per-target force routine, so every body's
// acceleration is accumulated in exactly the same order (j = 0..N-1, skipping
// j == i). OpenMP only changes which thread computes which target, so its
// results are expected to be bitwise identical to the serial reference.

#include "nbody/simulation.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace nbody::cpu_detail {

// Softened direct-sum acceleration on body i from every other body (G = 1).
// Mirrors the arithmetic of the CUDA kernels.
inline Acceleration acceleration_on(
    const Body* bodies,
    std::size_t count,
    std::size_t i,
    double epsilon_squared
) {
    const double target_x = bodies[i].x;
    const double target_y = bodies[i].y;
    const double target_z = bodies[i].z;

    double ax = 0.0;
    double ay = 0.0;
    double az = 0.0;

    for (std::size_t j = 0; j < count; ++j) {
        // skip self-interaction; required when epsilon == 0
        if (j == i) {
            continue;
        }

        const Body& source = bodies[j];

        // displacement from target to source
        const double dx = source.x - target_x;
        const double dy = source.y - target_y;
        const double dz = source.z - target_z;

        // 1 / (r^2 + eps^2)^(3/2)
        const double squared_softened_distance =
            dx * dx + dy * dy + dz * dz + epsilon_squared;
        const double inverse_distance = 1.0 / std::sqrt(squared_softened_distance);
        const double inverse_distance_cubed =
            inverse_distance * inverse_distance * inverse_distance;

        const double interaction_strength = source.mass * inverse_distance_cubed;
        ax += interaction_strength * dx;
        ay += interaction_strength * dy;
        az += interaction_strength * dz;
    }

    return Acceleration{ax, ay, az};
}

inline void validate_simulation_arguments(double dt, double epsilon) {
    if (dt <= 0.0) {
        throw std::invalid_argument("dt must be positive");
    }
    if (epsilon < 0.0) {
        throw std::invalid_argument("epsilon cannot be negative");
    }
}

// Velocity-Verlet integration with the same step structure as the CUDA
// backends:
//   a = F(x0)
//   repeat steps times:
//     v += 0.5 * a * dt;  x += v * dt     (kick + drift)
//     a = F(x)                            (new forces)
//     v += 0.5 * a * dt                   (final kick)
//
// compute_forces(bodies, accelerations) must overwrite every acceleration.
// The O(N) kick/drift loops stay serial; the O(N^2) force call dominates.
template <typename ComputeForces>
Bodies simulate_velocity_verlet(
    const Bodies& initial_bodies,
    std::size_t steps,
    double dt,
    double epsilon,
    ComputeForces&& compute_forces
) {
    Bodies bodies = initial_bodies;
    if (bodies.empty() || steps == 0) {
        return bodies;
    }

    validate_simulation_arguments(dt, epsilon);

    Accelerations accelerations(bodies.size());
    compute_forces(bodies, accelerations);

    for (std::size_t step = 0; step < steps; ++step) {
        for (std::size_t i = 0; i < bodies.size(); ++i) {
            Body& body = bodies[i];
            const Acceleration& acc = accelerations[i];

            body.vx += 0.5 * acc.x * dt;
            body.vy += 0.5 * acc.y * dt;
            body.vz += 0.5 * acc.z * dt;

            body.x += body.vx * dt;
            body.y += body.vy * dt;
            body.z += body.vz * dt;
        }

        compute_forces(bodies, accelerations);

        for (std::size_t i = 0; i < bodies.size(); ++i) {
            Body& body = bodies[i];
            const Acceleration& acc = accelerations[i];

            body.vx += 0.5 * acc.x * dt;
            body.vy += 0.5 * acc.y * dt;
            body.vz += 0.5 * acc.z * dt;
        }
    }

    return bodies;
}

}  // namespace nbody::cpu_detail
