#pragma once

#include "nbody/body.hpp"
#include "nbody/config.hpp"

#include <vector>

namespace nbody {

using Bodies = std::vector<Body>;
using Accelerations = std::vector<Acceleration>;

Bodies initialize_bodies(std::size_t count, std::uint64_t seed);

// Backend extension points. The force calculations are the next milestone.
Accelerations compute_acceleration_serial(const Bodies& bodies, double epsilon);
Accelerations compute_acceleration_openmp(const Bodies& bodies, double epsilon);

Accelerations compute_acceleration_cuda_naive(
    const Bodies& bodies,
    double epsilon,
    int block_size
);

Accelerations compute_acceleration_cuda_tiled(
    const Bodies& bodies,
    double epsilon,
    int block_size
);

}  // namespace nbody

