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
    // =============================================
    // 2 body test with Body 0 position: (-1, 0, 0), 
    //                  Body 1 position: (+1, 0, 0)
    // expect   Body 0 acceleration: (+0.25, 0, 0)
    //          Body 1 acceleration: (-0.25, 0, 0)
    // =============================================

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

    // ============================================================
    // 257 body test arranged in a (partially filled) 7 x 7 x 7 grid
    // with a spacing of 0.5 m apart centered at -3, -3. -2.5
    // ============================================================
    {
        nbody::Bodies test_bodies_257(257);

        // create 257 body structs
        for(std::size_t i = 0; i < test_bodies_257.size(); i++) {
            nbody::Body& body = test_bodies_257[i];

            body.x = (static_cast<double>(i % 7) - 3.0) * 0.5;
            body.y = (static_cast<double>((i / 7) % 7) - 3.0) * 0.5;
            body.z = (static_cast<double>((i / 49) % 7) - 2.5) * 0.5;

            body.vx = 0.0;
            body.vy = 0.0;
            body.vz = 0.0;
            body.mass = 1.0;
        }

        const double epsilon = 0.01;

        nbody::Accelerations tiled_acc_257 = nbody::compute_acceleration_cuda_tiled(test_bodies_257, epsilon, 256);
        nbody::Accelerations naive_acc_257 = nbody::compute_acceleration_cuda_naive(test_bodies_257, epsilon, 256);

        // check tiled acceleration size is 257
        if (tiled_acc_257.size() != 257) {
            std::cerr << "Expected 257 accelerations in tiled, but received "
                    << tiled_acc_257.size() << '\n';
            return 1;
        }

        // check naive acceleration size is 257
        if (naive_acc_257.size() != 257) {
            std::cerr << "Expected 257 accelerations in naive, but received "
                    << naive_acc_257.size() << '\n';
            return 1;
        }

        // check all tiled component is finite
        // and compare tiled with naive
        for(std::size_t i = 0; i < tiled_acc_257.size(); i++) {
            const bool finite_components = 
                std::isfinite(tiled_acc_257[i].x) && 
                std::isfinite(tiled_acc_257[i].y) && 
                std::isfinite(tiled_acc_257[i].z);
            
            if (!finite_components) {
                std::cerr
                    << "Non-finite tiled acceleration at body " << i << ": ("
                    << tiled_acc_257[i].x << ", "
                    << tiled_acc_257[i].y << ", "
                    << tiled_acc_257[i].z << ")\n";
                return 1;
            }

            const bool correct = 
                approximately_equal(tiled_acc_257[i].x, naive_acc_257[i].x, 1e-10) &&
                approximately_equal(tiled_acc_257[i].y, naive_acc_257[i].y, 1e-10) &&
                approximately_equal(tiled_acc_257[i].z, naive_acc_257[i].z, 1e-10);

            if (!correct) {
                std::cerr
                    << "Unexpected acceleration results:\n"
                    << "Body " << i << ":\n"
                    << "naive ("
                    << naive_acc_257[i].x << ", "
                    << naive_acc_257[i].y << ", "
                    << naive_acc_257[i].z << ")\n"
                    << "tiled ("
                    << tiled_acc_257[i].x << ", "
                    << tiled_acc_257[i].y << ", "
                    << tiled_acc_257[i].z << ")\n";

                return 1;
            }
        }
        
        std::cout << "CUDA tiled 257 test passed\n";
    }
    return 0;
}