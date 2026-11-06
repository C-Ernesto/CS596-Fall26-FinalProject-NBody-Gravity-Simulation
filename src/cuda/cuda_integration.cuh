#pragma once

#include "nbody/body.hpp"

#include <cuda_runtime.h>

#include <cstddef>

namespace nbody::cuda_detail {
    static __global__ void kick_drift_kernel(
        Body* bodies,
        const Acceleration* accelerations,
        std::size_t count,
        double dt
    ) {
        // calculate global body index
        const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

        if (i >= count) {
            return;
        }

        const Acceleration curr_acc = accelerations[i];
        Body& curr_body = bodies[i];

        // update velocity of body by half a step
        curr_body.vx += 0.5 * curr_acc.x * dt;
        curr_body.vy += 0.5 * curr_acc.y * dt;
        curr_body.vz += 0.5 * curr_acc.z * dt;

        // update position
        curr_body.x += curr_body.vx * dt;
        curr_body.y += curr_body.vy * dt;
        curr_body.z += curr_body.vz * dt;
    }
    
    static __global__ void final_kick_kernel(
        Body* bodies,
        const Acceleration* accelerations,
        std::size_t count,
        double dt
    ) {
        // calculate global body index
        const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

        if (i >= count) {
            return;
        }

        Body& curr_body = bodies[i];
        const Acceleration curr_acc = accelerations[i];

        // Add the remaining half velocity step
        curr_body.vx += 0.5 * curr_acc.x * dt;
        curr_body.vy += 0.5 * curr_acc.y * dt;
        curr_body.vz += 0.5 * curr_acc.z * dt;
    }

}  // namespace nbody::cuda_detail