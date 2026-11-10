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

    // ============================================================
    // 1 body motion test 
    // with Initial position: (1, 2, 3), Initial velocity: (0.5, -0.25, 1.0)
    // Mass: 1, Steps: 10, dt: 0.1
    // expect: Final position: (1.5, 1.75, 4.0)
    // Final velocity: (0.5, -0.25, 1.0)
    // ============================================================
    {
        const nbody::Bodies initial_bodies = {{1.0, 2.0, 3.0, 0.5, -0.25, 1.0, 1.0}};
        const std::size_t steps = 10;
        const double dt = 0.1;
        nbody::CudaTiming timing{};

        const nbody::Bodies test_body_single = nbody::simulate_cuda_tiled(
            initial_bodies,
            steps,
            dt,
            0.01,
            256,
            &timing
        );

        const double measured_components =
            timing.allocation_ms + timing.host_to_device_ms + timing.kernel_ms +
            timing.device_to_host_ms + timing.cleanup_ms;
        if (timing.allocation_ms <= 0.0 || timing.host_to_device_ms <= 0.0 ||
            timing.kernel_ms <= 0.0 || timing.device_to_host_ms <= 0.0 ||
            timing.cleanup_ms <= 0.0 || timing.total_ms < measured_components) {
            std::cerr << "CUDA tiled timing breakdown is invalid\n";
            return 1;
        }

        // check only 1 body is returned
        if (test_body_single.size() != 1) {
            std::cerr << "Expected 1 body, but received "
                    << test_body_single.size() << '\n';
            return 1;
        }

        const bool single_correct = approximately_equal(test_body_single[0].x, 1.5, 1e-12) &&
                            approximately_equal(test_body_single[0].y, 1.75, 1e-12) &&
                            approximately_equal(test_body_single[0].z, 4.0, 1e-12) &&
                            approximately_equal(test_body_single[0].vx, 0.5, 1e-12) &&
                            approximately_equal(test_body_single[0].vy, -0.25, 1e-12) &&
                            approximately_equal(test_body_single[0].vz, 1.0, 1e-12) &&
                            approximately_equal(test_body_single[0].mass, 1.0, 1e-12);

        if (!single_correct) {
            std::cerr
                << "Unexpected final body state:\n"
                << "Body: ("
                << test_body_single[0].x << ", "
                << test_body_single[0].y << ", "
                << test_body_single[0].z << ", "
                << test_body_single[0].vx << ", "
                << test_body_single[0].vy << ", "
                << test_body_single[0].vz << ", "
                << test_body_single[0].mass << ")\n";

            return 1;
        }

        std::cout << "CUDA tiled one-body motion test passed\n";
    }

    // ============================================================
    // 2 body motion test 
    // Body 1 = Initial position: (-1, 0, 0), Initial velocity: (0, 0, 0), Mass: 1
    // Body 2 = Initial position: (1, 0, 0), Initial velocity: (0, 0, 0), Mass: 1
    // Steps: 1, dt: 0.1
    // expect: 
    // initial acceleration magnitude = 0.25
    // half-step velocity = 0.5 × 0.25 × dt
    // new left position = -1 + half-step velocity × dt
    // new right position = +1 - half-step velocity × dt
    // new separation = right position - left position
    // new acceleration magnitude = 1 / separation²
    // final velocity magnitude = half-step velocity + 0.5 × new acceleration × dt
    // ============================================================
    {
        const nbody::Bodies initial_bodies = {
            {-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0},
            { 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}
        };

        const std::size_t steps = 1;
        const double dt = 0.1;
        const double epsilon = 0.0;

        const nbody::Bodies final_bodies = nbody::simulate_cuda_tiled(initial_bodies, steps, dt, epsilon, 256);

        if (final_bodies.size() != 2) {
            std::cerr
                << "Expected 2 final bodies, but received "
                << final_bodies.size() << '\n';

            return 1;
        }

        // Initial acceleration magnitude:
        // mass / distance^2 = 1 / 2^2 = 0.25
        const double initial_acceleration = 0.25;

        // First half of the velocity-Verlet velocity update.
        const double half_velocity =
            0.5 * initial_acceleration * dt;

        // Positions after the kick-drift kernel.
        const double expected_left_x =
            -1.0 + half_velocity * dt;

        const double expected_right_x =
            1.0 - half_velocity * dt;

        // Recalculate acceleration at the new positions.
        const double new_separation =
            expected_right_x - expected_left_x;

        const double new_acceleration =
            1.0 / (new_separation * new_separation);

        // Complete the second half of the velocity update.
        const double expected_speed =
            half_velocity + 0.5 * new_acceleration * dt;

        const bool correct =
            approximately_equal(
                final_bodies[0].x,
                expected_left_x,
                1e-12
            ) &&
            approximately_equal(final_bodies[0].y, 0.0, 1e-12) &&
            approximately_equal(final_bodies[0].z, 0.0, 1e-12) &&
            approximately_equal(
                final_bodies[0].vx,
                expected_speed,
                1e-12
            ) &&
            approximately_equal(final_bodies[0].vy, 0.0, 1e-12) &&
            approximately_equal(final_bodies[0].vz, 0.0, 1e-12) &&

            approximately_equal(
                final_bodies[1].x,
                expected_right_x,
                1e-12
            ) &&
            approximately_equal(final_bodies[1].y, 0.0, 1e-12) &&
            approximately_equal(final_bodies[1].z, 0.0, 1e-12) &&
            approximately_equal(
                final_bodies[1].vx,
                -expected_speed,
                1e-12
            ) &&
            approximately_equal(final_bodies[1].vy, 0.0, 1e-12) &&
            approximately_equal(final_bodies[1].vz, 0.0, 1e-12) &&

            approximately_equal(final_bodies[0].mass, 1.0, 1e-12) &&
            approximately_equal(final_bodies[1].mass, 1.0, 1e-12);

        if (!correct) {
            std::cerr
                << "Unexpected two-body simulation result:\n"
                << "Expected left x/vx: "
                << expected_left_x << ", "
                << expected_speed << '\n'
                << "Actual left x/vx: "
                << final_bodies[0].x << ", "
                << final_bodies[0].vx << '\n'
                << "Expected right x/vx: "
                << expected_right_x << ", "
                << -expected_speed << '\n'
                << "Actual right x/vx: "
                << final_bodies[1].x << ", "
                << final_bodies[1].vx << '\n';

            return 1;
        }

        std::cout
            << "CUDA tiled two-body simulation test passed\n";
    }

    // ============================================================
    // 257 body test arranged in a (partially filled) 7 x 7 x 7 grid
    // with a spacing of 0.5 m apart centered at -3, -3. -2.5
    // ============================================================
    {
        nbody::Bodies initial_bodies_257(257);

        // create 257 body structs
        for(std::size_t i = 0; i < initial_bodies_257.size(); i++) {
            nbody::Body& body = initial_bodies_257[i];

            body.x = (static_cast<double>(i % 7) - 3.0) * 0.5;
            body.y = (static_cast<double>((i / 7) % 7) - 3.0) * 0.5;
            body.z = (static_cast<double>((i / 49) % 7) - 2.5) * 0.5;

            body.vx = 0.0;
            body.vy = 0.0;
            body.vz = 0.0;
            body.mass = 1.0;
        }

        const double epsilon = 0.01;
        const std::size_t steps = 5;
        const double dt = 0.001;
        const int block_size = 256;

        const nbody::Bodies final_bodies_naive = nbody::simulate_cuda_naive(initial_bodies_257, steps, dt, epsilon, block_size);
        const nbody::Bodies final_bodies_tiled = nbody::simulate_cuda_tiled(initial_bodies_257, steps, dt, epsilon, block_size);

        // check naive bodies size is 257
        if (final_bodies_naive.size() != 257) {
            std::cerr << "Expected 257 bodies in naive, but received "
                    << final_bodies_naive.size() << '\n';
            return 1;
        }

        // check tiled bodies size is 257
        if (final_bodies_tiled.size() != 257) {
            std::cerr << "Expected 257 bodies in tiled, but received "
                    << final_bodies_tiled.size() << '\n';
            return 1;
        }

        // check all tiled component is finite
        // and compare tiled with naive
        for(std::size_t i = 0; i < final_bodies_tiled.size(); i++) {
            const bool finite_components = 
                std::isfinite(final_bodies_tiled[i].x) && 
                std::isfinite(final_bodies_tiled[i].y) && 
                std::isfinite(final_bodies_tiled[i].z) &&
                std::isfinite(final_bodies_tiled[i].vx) &&
                std::isfinite(final_bodies_tiled[i].vy) &&
                std::isfinite(final_bodies_tiled[i].vz);
            
            if (!finite_components) {
                std::cerr
                    << "Non-finite tiled body at body " << i << ": pos = ("
                    << final_bodies_tiled[i].x << ", "
                    << final_bodies_tiled[i].y << ", "
                    << final_bodies_tiled[i].z << ")\n"
                    << "vel = ("
                    << final_bodies_tiled[i].vx << ", "
                    << final_bodies_tiled[i].vy << ", "
                    << final_bodies_tiled[i].vz << ")\n";
                return 1;
            }

            const bool correct = 
                approximately_equal(final_bodies_tiled[i].x, final_bodies_naive[i].x, 1e-10) &&
                approximately_equal(final_bodies_tiled[i].y, final_bodies_naive[i].y, 1e-10) &&
                approximately_equal(final_bodies_tiled[i].z, final_bodies_naive[i].z, 1e-10) &&
                approximately_equal(final_bodies_tiled[i].vx, final_bodies_naive[i].vx, 1e-10) &&
                approximately_equal(final_bodies_tiled[i].vy, final_bodies_naive[i].vy, 1e-10) &&
                approximately_equal(final_bodies_tiled[i].vz, final_bodies_naive[i].vz, 1e-10);

            if (!correct) {
                std::cerr
                    << "Unexpected final body states:\n"
                    << "Body " << i << ":\n"
                    << "naive ("
                    << final_bodies_naive[i].x << ", "
                    << final_bodies_naive[i].y << ", "
                    << final_bodies_naive[i].z << ", "
                    << final_bodies_naive[i].vx << ", "
                    << final_bodies_naive[i].vy << ", "
                    << final_bodies_naive[i].vz << ")\n"
                    << "tiled ("
                    << final_bodies_tiled[i].x << ", "
                    << final_bodies_tiled[i].y << ", "
                    << final_bodies_tiled[i].z << ", "
                    << final_bodies_tiled[i].vx << ", "
                    << final_bodies_tiled[i].vy << ", "
                    << final_bodies_tiled[i].vz << ")\n";

                return 1;
            }

            // check mass is unchanged
            const bool mass_check = 
                approximately_equal(final_bodies_naive[i].mass, initial_bodies_257[i].mass, 1e-12) && 
                approximately_equal(final_bodies_tiled[i].mass, initial_bodies_257[i].mass, 1e-12);

            if (!mass_check) {
                std::cerr
                    << "Unexpected mass:\n"
                    << "Body " << i << ":\n"
                    << "naive : " << final_bodies_naive[i].mass << "\n"
                    << "tiled : " << final_bodies_tiled[i].mass << "\n";

                return 1;
            }
        }
        
        std::cout << "CUDA tiled 257 body simulation passed\n";
    }

    return 0;
} // end of main
