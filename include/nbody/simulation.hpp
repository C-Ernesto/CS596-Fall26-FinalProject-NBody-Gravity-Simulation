#pragma once

#include "nbody/body.hpp"
#include "nbody/config.hpp"

#include <vector>

namespace nbody {

using Bodies = std::vector<Body>;
using Accelerations = std::vector<Acceleration>;

struct CudaTiming {
    double allocation_ms{};
    double host_to_device_ms{};
    double kernel_ms{};
    double device_to_host_ms{};
    double cleanup_ms{};
    double total_ms{};
};

Bodies initialize_bodies(std::size_t count, std::uint64_t seed);

// Exact FP64 CPU backends. The serial backend is the independent reference.
Accelerations compute_acceleration_serial(const Bodies& bodies, double epsilon);

Bodies simulate_serial(
    const Bodies& initial_bodies,
    std::size_t steps,
    double dt,
    double epsilon
);

// threads == 0 uses the OpenMP default (OMP_NUM_THREADS or all hardware threads).
Accelerations compute_acceleration_openmp(
    const Bodies& bodies,
    double epsilon,
    int threads = 0
);

Bodies simulate_openmp(
    const Bodies& initial_bodies,
    std::size_t steps,
    double dt,
    double epsilon,
    int threads = 0
);

// Number of threads the OpenMP backend will actually use for a request.
int openmp_thread_count(int requested_threads);

Accelerations compute_acceleration_cuda_naive(
    const Bodies& bodies,
    double epsilon,
    int block_size
);

Accelerations compute_acceleration_cuda_tiled(
    const Bodies& bodies,
    double epsilon,
    int block_size,
    CudaTiming* timing = nullptr
);

Bodies simulate_cuda_naive(
    const Bodies& initial_bodies,
    std::size_t steps,
    double dt,
    double epsilon,
    int block_size,
    CudaTiming* timing = nullptr
);

Bodies simulate_cuda_tiled(
    const Bodies& initial_bodies,
    std::size_t steps,
    double dt,
    double epsilon,
    int block_size,
    CudaTiming* timing = nullptr
);

}  // namespace nbody

