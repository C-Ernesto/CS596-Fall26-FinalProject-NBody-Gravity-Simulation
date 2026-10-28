#include "nbody/simulation.hpp"

#include <cmath>
#include <iostream>

bool approximately_equal(double actual, double expected, double tolerance) {
    return std::abs(actual - expected) < tolerance;
}

int main() {

    
    // =============================================
    // 1 body test with Body 0 position: (3, -2, 5)
    // expect   Body 0 acceleration: (0, 0, 0)
    // =============================================
    {
        nbody::Bodies test_body_single = {{3.0, -2.0, 5.0, 0.0, 0.0, 0.0, 2.0}};

        nbody::Accelerations test_acc_single = nbody::compute_acceleration_cuda_naive(test_body_single, 0.01, 256);

        //check only 1 acceleration is returned
        if (test_acc_single.size() != 1) {
            std::cerr << "Expected 1 acceleration, but received "
                    << test_acc_single.size() << '\n';
            return 1;
        }

        const bool single_correct = approximately_equal(test_acc_single[0].x, 0, 1e-12) &&
                            approximately_equal(test_acc_single[0].y, 0, 1e-12) &&
                            approximately_equal(test_acc_single[0].z, 0, 1e-12);

        if (!single_correct) {
            std::cerr
                << "Unexpected acceleration results:\n"
                << "Body: ("
                << test_acc_single[0].x << ", "
                << test_acc_single[0].y << ", "
                << test_acc_single[0].z << ")\n";

            return 1;
        }

        std::cout << "CUDA naive one-body test passed\n";
    }

    // =============================================
    // 2 body test with Body 0 position: (-1, 0, 0), 
    //                  Body 1 position: (+1, 0, 0)
    // expect   Body 0 acceleration: (+0.25, 0, 0)
    //          Body 1 acceleration: (-0.25, 0, 0)
    // =============================================
    {
        nbody::Bodies test_bodies = {
            {-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0},
            {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}
        };

        const double epsilon = 0.0;

        nbody::Accelerations test_acc = nbody::compute_acceleration_cuda_naive(test_bodies, epsilon, 256);

        // check acceleration size is 2
        if (test_acc.size() != 2) {
            std::cerr << "Expected 2 accelerations, but received "
                    << test_acc.size() << '\n';
            return 1;
        }

        const bool two_body_correct = approximately_equal(test_acc[0].x, 0.25, 1e-12) &&
                            approximately_equal(test_acc[0].y, 0, 1e-12) &&
                            approximately_equal(test_acc[0].z, 0, 1e-12) &&
                            approximately_equal(test_acc[1].x, -0.25, 1e-12) &&
                            approximately_equal(test_acc[1].y, 0, 1e-12) &&
                            approximately_equal(test_acc[1].z, 0, 1e-12);

        if (!two_body_correct) {
            std::cerr
                << "Unexpected acceleration results:\n"
                << "Body 0: ("
                << test_acc[0].x << ", "
                << test_acc[0].y << ", "
                << test_acc[0].z << ")\n"
                << "Body 1: ("
                << test_acc[1].x << ", "
                << test_acc[1].y << ", "
                << test_acc[1].z << ")\n";

            return 1;
        }
        
        std::cout << "CUDA naive two-body test passed\n";
    }

    // ============================================================
    // 257 body test arranged in a (partially filled) 7 x 7 x 7 grid
    // with a spacing of 0.5 m apart centered at -3, -3. -2.5
    // expect: total sum of acceleration of all bodies to be 0
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

        nbody::Accelerations test_acc_257 = nbody::compute_acceleration_cuda_naive(test_bodies_257, epsilon, 256);

        // check acceleration size is 257
        if (test_acc_257.size() != 257) {
            std::cerr << "Expected 257 accelerations, but received "
                    << test_acc_257.size() << '\n';
            return 1;
        }

        double total_ax = 0.0, total_ay = 0.0, total_az = 0.0;

        for (const nbody::Acceleration& acc : test_acc_257) {
            // check each acceleration value is finite
            if (!std::isfinite(acc.x) || !std::isfinite(acc.y) || !std::isfinite(acc.z)) {
                std::cerr << "Found a non-finite acceleration\n";
                return 1;
            }

            //sum all accelerations components
            total_ax += acc.x;
            total_ay += acc.y;
            total_az += acc.z;

        }

        // check all accelerations totals are 0
        const bool momentum_conserved = approximately_equal(total_ax, 0.0, 1e-9) &&
                                        approximately_equal(total_ay, 0.0, 1e-9) &&
                                        approximately_equal(total_az, 0.0, 1e-9);

        if (!momentum_conserved) {
            std::cerr
                << "Unexpected acceleration results:\n"
                << "total acceleration: ("
                << total_ax << ", "
                << total_ay << ", "
                << total_az << ")\n";

            return 1;
        }
        
        std::cout << "CUDA naive 257 test passed\n";

    }

    return 0;
}