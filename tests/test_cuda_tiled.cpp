#include "nbody/simulation.hpp"

#include <cmath>
#include <iostream>

bool approximately_equal(
    double actual,
    double expected,
    double tolerance = 1e-12
) {
    return std::abs(actual - expected) < tolerance;
}

int main() {
    // Two equal-mass bodies separated by a distance of 2.
    nbody::Bodies test_bodies = {
        {-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0},
        { 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}
    };

    const nbody::Accelerations accelerations =
        nbody::compute_acceleration_cuda_tiled(
            test_bodies,
            0.0,
            256
        );

    if (accelerations.size() != 2) {
        std::cerr
            << "Expected 2 accelerations, but received "
            << accelerations.size() << '\n';

        return 1;
    }

    const bool correct =
        approximately_equal(accelerations[0].x, 0.25) &&
        approximately_equal(accelerations[0].y, 0.0) &&
        approximately_equal(accelerations[0].z, 0.0) &&
        approximately_equal(accelerations[1].x, -0.25) &&
        approximately_equal(accelerations[1].y, 0.0) &&
        approximately_equal(accelerations[1].z, 0.0);

    if (!correct) {
        std::cerr
            << "Unexpected tiled CUDA acceleration results:\n"
            << "Body 0: ("
            << accelerations[0].x << ", "
            << accelerations[0].y << ", "
            << accelerations[0].z << ")\n"
            << "Body 1: ("
            << accelerations[1].x << ", "
            << accelerations[1].y << ", "
            << accelerations[1].z << ")\n";

        return 1;
    }

    std::cout << "CUDA tiled two-body test passed\n";
    return 0;
}