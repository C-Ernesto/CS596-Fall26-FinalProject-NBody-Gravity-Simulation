// Shared-memory parallel CPU backend using OpenMP.
//
// The outer loop over target bodies is split across threads. Each thread
// writes only its own targets' accelerations, so no locks or reductions are
// needed, and each target is summed in the same order as the serial backend.
// Results are therefore expected to match serial bit for bit.

#include "nbody/simulation.hpp"

#include "cpu_detail.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace nbody {

namespace {

void compute_openmp_into(
    const Bodies& bodies,
    Accelerations& accelerations,
    double epsilon_squared,
    int threads
) {
#ifdef _OPENMP
    const Body* data = bodies.data();
    const std::size_t count = bodies.size();

    // MSVC's default /openmp implements OpenMP 2.0, which requires a signed
    // loop index for "parallel for".
    const std::int64_t signed_count = static_cast<std::int64_t>(count);

    // Static scheduling: every target costs the same N-1 interactions.
#pragma omp parallel for num_threads(threads) schedule(static)
    for (std::int64_t i = 0; i < signed_count; ++i) {
        const std::size_t target = static_cast<std::size_t>(i);
        accelerations[target] =
            cpu_detail::acceleration_on(data, count, target, epsilon_squared);
    }
#else
    (void)bodies;
    (void)accelerations;
    (void)epsilon_squared;
    (void)threads;
    throw std::runtime_error("OpenMP support is not enabled in this build");
#endif
}

}  // namespace

int openmp_thread_count(int requested_threads) {
    if (requested_threads < 0) {
        throw std::invalid_argument("thread count cannot be negative");
    }
    if (requested_threads > 0) {
        return requested_threads;
    }
#ifdef _OPENMP
    // 0 means "use the OpenMP default" (OMP_NUM_THREADS or all hardware threads)
    return omp_get_max_threads();
#else
    return 1;
#endif
}

Accelerations compute_acceleration_openmp(
    const Bodies& bodies,
    double epsilon,
    int threads
) {
    if (epsilon < 0.0) {
        throw std::invalid_argument("epsilon cannot be negative");
    }

    const int thread_count = openmp_thread_count(threads);
    Accelerations accelerations(bodies.size());
    compute_openmp_into(bodies, accelerations, epsilon * epsilon, thread_count);
    return accelerations;
}

Bodies simulate_openmp(
    const Bodies& initial_bodies,
    std::size_t steps,
    double dt,
    double epsilon,
    int threads
) {
    const int thread_count = openmp_thread_count(threads);
    const double epsilon_squared = epsilon * epsilon;
    return cpu_detail::simulate_velocity_verlet(
        initial_bodies,
        steps,
        dt,
        epsilon,
        [epsilon_squared, thread_count](
            const Bodies& bodies,
            Accelerations& accelerations
        ) {
            compute_openmp_into(bodies, accelerations, epsilon_squared, thread_count);
        }
    );
}

}  // namespace nbody
