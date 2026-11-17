// Serial CPU reference backend: one thread, exact O(N^2) direct summation in
// FP64. This is the independent reference used to validate every other
// backend, so it is intentionally kept simple rather than optimized.

#include "nbody/simulation.hpp"

#include "cpu_detail.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace nbody {

namespace {

void compute_serial_into(
    const Bodies& bodies,
    Accelerations& accelerations,
    double epsilon_squared
) {
    const Body* data = bodies.data();
    const std::size_t count = bodies.size();

    for (std::size_t i = 0; i < count; ++i) {
        accelerations[i] = cpu_detail::acceleration_on(data, count, i, epsilon_squared);
    }
}

}  // namespace

Accelerations compute_acceleration_serial(const Bodies& bodies, double epsilon) {
    if (!std::isfinite(epsilon) || epsilon < 0.0) {
        throw std::invalid_argument("epsilon must be finite and nonnegative");
    }

    Accelerations accelerations(bodies.size());
    compute_serial_into(bodies, accelerations, epsilon * epsilon);
    return accelerations;
}

Bodies simulate_serial(
    const Bodies& initial_bodies,
    std::size_t steps,
    double dt,
    double epsilon
) {
    const double epsilon_squared = epsilon * epsilon;
    return cpu_detail::simulate_velocity_verlet(
        initial_bodies,
        steps,
        dt,
        epsilon,
        [epsilon_squared](const Bodies& bodies, Accelerations& accelerations) {
            compute_serial_into(bodies, accelerations, epsilon_squared);
        }
    );
}

}  // namespace nbody
