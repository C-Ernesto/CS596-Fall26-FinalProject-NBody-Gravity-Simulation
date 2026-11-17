// Correctness tests for the serial and OpenMP CPU backends.
//
// - analytical one- and two-body forces
// - zero net force for a 257-body grid (Newton's third law)
// - OpenMP accelerations and trajectories bitwise equal to serial
// - velocity-Verlet energy conservation on a circular two-body orbit
// - optional serial-vs-CUDA agreement when CUDA is enabled

#include "nbody/simulation.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using ForceFunction = std::function<nbody::Accelerations(const nbody::Bodies&, double)>;

struct NamedForce {
    std::string name;
    ForceFunction compute;
};

int failures = 0;

void check(bool condition, const std::string& message) {
    if (condition) {
        std::cout << "  passed: " << message << '\n';
    } else {
        std::cerr << "  FAILED: " << message << '\n';
        ++failures;
    }
}

bool approximately_equal(double actual, double expected, double tolerance) {
    return std::abs(actual - expected) < tolerance;
}

nbody::Bodies grid_257() {
    nbody::Bodies bodies(257);
    for (std::size_t i = 0; i < bodies.size(); ++i) {
        nbody::Body& body = bodies[i];
        body.x = (static_cast<double>(i % 7) - 3.0) * 0.5;
        body.y = (static_cast<double>((i / 7) % 7) - 3.0) * 0.5;
        body.z = (static_cast<double>((i / 49) % 7) - 2.5) * 0.5;
        body.mass = 1.0;
    }
    return bodies;
}

bool bitwise_equal(const nbody::Accelerations& a, const nbody::Accelerations& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i].x != b[i].x || a[i].y != b[i].y || a[i].z != b[i].z) return false;
    }
    return true;
}

bool bitwise_equal(const nbody::Bodies& a, const nbody::Bodies& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i].x != b[i].x || a[i].y != b[i].y || a[i].z != b[i].z ||
            a[i].vx != b[i].vx || a[i].vy != b[i].vy || a[i].vz != b[i].vz ||
            a[i].mass != b[i].mass) {
            return false;
        }
    }
    return true;
}

// max_i |a_i - b_i| / max_i |b_i|
[[maybe_unused]] double relative_max_error(const nbody::Accelerations& a, const nbody::Accelerations& b) {
    double max_difference = 0.0;
    double max_magnitude = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        max_difference = std::max({max_difference,
            std::abs(a[i].x - b[i].x), std::abs(a[i].y - b[i].y), std::abs(a[i].z - b[i].z)});
        max_magnitude = std::max({max_magnitude,
            std::abs(b[i].x), std::abs(b[i].y), std::abs(b[i].z)});
    }
    return max_magnitude > 0.0 ? max_difference / max_magnitude : max_difference;
}

// Total energy consistent with the softened force law (G = 1).
double total_energy(const nbody::Bodies& bodies, double epsilon) {
    double kinetic = 0.0;
    double potential = 0.0;
    for (std::size_t i = 0; i < bodies.size(); ++i) {
        const nbody::Body& a = bodies[i];
        kinetic += 0.5 * a.mass * (a.vx * a.vx + a.vy * a.vy + a.vz * a.vz);
        for (std::size_t j = i + 1; j < bodies.size(); ++j) {
            const nbody::Body& b = bodies[j];
            const double dx = b.x - a.x, dy = b.y - a.y, dz = b.z - a.z;
            potential -= a.mass * b.mass /
                std::sqrt(dx * dx + dy * dy + dz * dz + epsilon * epsilon);
        }
    }
    return kinetic + potential;
}

std::vector<NamedForce> cpu_backends() {
    std::vector<NamedForce> backends{
        {"serial", [](const nbody::Bodies& b, double e) {
            return nbody::compute_acceleration_serial(b, e);
        }},
    };
#ifdef NBODY_HAS_OPENMP
    backends.push_back({"openmp", [](const nbody::Bodies& b, double e) {
        return nbody::compute_acceleration_openmp(b, e, 4);
    }});
#endif
    return backends;
}

void test_analytical_forces(const NamedForce& backend) {
    std::cout << backend.name << " analytical forces\n";

    // a single body feels no force
    {
        const nbody::Bodies bodies = {{3.0, -2.0, 5.0, 0.0, 0.0, 0.0, 2.0}};
        const nbody::Accelerations acc = backend.compute(bodies, 0.01);
        check(acc.size() == 1 && acc[0].x == 0.0 && acc[0].y == 0.0 && acc[0].z == 0.0,
              "one body has zero acceleration");
    }

    // two unit masses at x = -1 and x = +1: |a| = 1 / 2^2 = 0.25, pointing inward
    {
        const nbody::Bodies bodies = {
            {-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0},
            {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0},
        };
        const nbody::Accelerations acc = backend.compute(bodies, 0.0);
        check(acc.size() == 2 &&
              approximately_equal(acc[0].x, 0.25, 1e-12) &&
              approximately_equal(acc[0].y, 0.0, 1e-12) &&
              approximately_equal(acc[0].z, 0.0, 1e-12) &&
              approximately_equal(acc[1].x, -0.25, 1e-12) &&
              approximately_equal(acc[1].y, 0.0, 1e-12) &&
              approximately_equal(acc[1].z, 0.0, 1e-12),
              "two-body accelerations are +/-0.25 with epsilon = 0");
    }

    // softening: |a| = m * d / (d^2 + eps^2)^(3/2) with d = 2, eps = 1
    {
        const nbody::Bodies bodies = {
            {-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0},
            {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0},
        };
        const double expected = 2.0 / std::pow(5.0, 1.5);
        const nbody::Accelerations acc = backend.compute(bodies, 1.0);
        check(approximately_equal(acc[0].x, expected, 1e-12) &&
              approximately_equal(acc[1].x, -expected, 1e-12),
              "softened two-body acceleration matches analytical value");
    }

    // 257-body grid: equal masses, so accelerations must sum to zero
    {
        const nbody::Accelerations acc = backend.compute(grid_257(), 0.01);
        bool finite = acc.size() == 257;
        double total_x = 0.0, total_y = 0.0, total_z = 0.0;
        for (const nbody::Acceleration& a : acc) {
            finite = finite && std::isfinite(a.x) && std::isfinite(a.y) && std::isfinite(a.z);
            total_x += a.x;
            total_y += a.y;
            total_z += a.z;
        }
        check(finite, "257-body accelerations are finite");
        check(approximately_equal(total_x, 0.0, 1e-9) &&
              approximately_equal(total_y, 0.0, 1e-9) &&
              approximately_equal(total_z, 0.0, 1e-9),
              "257-body net acceleration is zero");
    }

    // empty input
    check(backend.compute(nbody::Bodies{}, 0.01).empty(), "empty input gives empty output");
}

void test_circular_orbit() {
    std::cout << "serial velocity-Verlet circular orbit\n";

    // two masses of 0.5 separated by 1 orbit their center at r = 0.5.
    // circular speed: v^2 / r = m / d^2  ->  v = 0.5, period T = 2*pi*r / v = 2*pi
    const double pi = std::acos(-1.0);
    const nbody::Bodies initial = {
        {-0.5, 0.0, 0.0, 0.0, -0.5, 0.0, 0.5},
        {0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.5},
    };
    const double dt = 0.001;
    const auto steps = static_cast<std::size_t>(std::llround(2.0 * pi / dt));

    const nbody::Bodies final_bodies = nbody::simulate_serial(initial, steps, dt, 0.0);

    const double e0 = total_energy(initial, 0.0);
    const double e1 = total_energy(final_bodies, 0.0);
    const double drift = std::abs((e1 - e0) / e0);
    check(drift < 1e-6, "energy drift over one orbit < 1e-6 (got " + std::to_string(drift) + ")");

    const double dx = final_bodies[1].x - final_bodies[0].x;
    const double dy = final_bodies[1].y - final_bodies[0].y;
    const double separation = std::sqrt(dx * dx + dy * dy);
    check(approximately_equal(separation, 1.0, 1e-5), "separation stays 1 over one orbit");

    check(approximately_equal(final_bodies[0].x, -0.5, 1e-3) &&
          approximately_equal(final_bodies[0].y, 0.0, 1e-3),
          "body returns to its starting point after one period");

    const double momentum_x = 0.5 * (final_bodies[0].vx + final_bodies[1].vx);
    const double momentum_y = 0.5 * (final_bodies[0].vy + final_bodies[1].vy);
    check(approximately_equal(momentum_x, 0.0, 1e-12) &&
          approximately_equal(momentum_y, 0.0, 1e-12),
          "total momentum remains zero");
}

void test_simulation_edge_cases() {
    std::cout << "serial simulation edge cases\n";
    const nbody::Bodies initial = nbody::initialize_bodies(16, 7);

    check(bitwise_equal(nbody::simulate_serial(initial, 0, 0.001, 0.01), initial),
          "zero steps returns the initial state");

    bool threw = false;
    try {
        (void)nbody::simulate_serial(initial, 1, 0.0, 0.01);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "non-positive dt is rejected");

    threw = false;
    try {
        (void)nbody::simulate_serial(
            initial,
            0,
            std::numeric_limits<double>::quiet_NaN(),
            0.01);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "non-finite dt is rejected even for zero steps");

    threw = false;
    try {
        (void)nbody::compute_acceleration_serial(
            initial, std::numeric_limits<double>::infinity());
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "non-finite epsilon is rejected");
}

#ifdef NBODY_HAS_OPENMP
void test_openmp_matches_serial() {
    std::cout << "OpenMP vs serial\n";

    // 1000 is not a multiple of most thread counts, so uneven splits are exercised
    const nbody::Bodies bodies = nbody::initialize_bodies(1000, 12345);
    const nbody::Accelerations reference = nbody::compute_acceleration_serial(bodies, 0.01);

    for (const int threads : {1, 2, 3, 4, 7, 0}) {
        const nbody::Accelerations acc = nbody::compute_acceleration_openmp(bodies, 0.01, threads);
        check(bitwise_equal(acc, reference),
              "accelerations bitwise equal to serial with threads = " + std::to_string(threads));
    }

    const nbody::Bodies serial_final = nbody::simulate_serial(bodies, 20, 0.001, 0.01);
    const nbody::Bodies openmp_final = nbody::simulate_openmp(bodies, 20, 0.001, 0.01, 4);
    check(bitwise_equal(openmp_final, serial_final),
          "20-step trajectory bitwise equal to serial");

    const int actual_threads = nbody::openmp_thread_count(3);
    check(actual_threads >= 1 && actual_threads <= 3,
          "reported OpenMP thread count is the observed team size");
    check(nbody::openmp_thread_count(0) >= 1, "default thread count is positive");

    bool threw = false;
    try {
        (void)nbody::compute_acceleration_openmp(bodies, 0.01, -1);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "negative thread count is rejected");
}
#endif

#ifdef NBODY_HAS_CUDA
void test_cuda_matches_serial() {
    std::cout << "CUDA vs serial (independent FP64 reference)\n";

    // GPU fused multiply-add changes rounding slightly, so compare with a tolerance
    const nbody::Bodies bodies = nbody::initialize_bodies(1000, 12345);
    const nbody::Accelerations reference = nbody::compute_acceleration_serial(bodies, 0.01);

    const double naive_error = relative_max_error(
        nbody::compute_acceleration_cuda_naive(bodies, 0.01, 64), reference);
    check(naive_error < 1e-12,
          "naive CUDA within 1e-12 of serial (got " + std::to_string(naive_error) + ")");

    const double tiled_error = relative_max_error(
        nbody::compute_acceleration_cuda_tiled(bodies, 0.01, 64), reference);
    check(tiled_error < 1e-12,
          "tiled CUDA within 1e-12 of serial (got " + std::to_string(tiled_error) + ")");

    const nbody::Bodies serial_final = nbody::simulate_serial(bodies, 10, 0.001, 0.01);
    const nbody::Bodies cuda_final = nbody::simulate_cuda_tiled(bodies, 10, 0.001, 0.01, 64);
    double max_position_error = 0.0;
    for (std::size_t i = 0; i < bodies.size(); ++i) {
        max_position_error = std::max({max_position_error,
            std::abs(serial_final[i].x - cuda_final[i].x),
            std::abs(serial_final[i].y - cuda_final[i].y),
            std::abs(serial_final[i].z - cuda_final[i].z)});
    }
    check(max_position_error < 1e-12, "10-step tiled CUDA trajectory matches serial");
}
#endif

}  // namespace

int main() {
    try {
        for (const NamedForce& backend : cpu_backends()) {
            test_analytical_forces(backend);
        }
        test_circular_orbit();
        test_simulation_edge_cases();
#ifdef NBODY_HAS_OPENMP
        test_openmp_matches_serial();
#else
        std::cout << "OpenMP not enabled; skipping OpenMP tests\n";
#endif
#ifdef NBODY_HAS_CUDA
        test_cuda_matches_serial();
#endif
    } catch (const std::exception& error) {
        std::cerr << "unexpected exception: " << error.what() << '\n';
        return 1;
    }

    if (failures != 0) {
        std::cerr << failures << " CPU test(s) failed\n";
        return 1;
    }
    std::cout << "all CPU tests passed\n";
    return 0;
}
