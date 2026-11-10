// Milestone placeholder: source bodies will be loaded in shared-memory tiles.

#include "nbody/simulation.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>
#include <string>
#include <stdexcept>
#include "cuda_integration.cuh"

namespace nbody {
    namespace {
        void check_cuda(cudaError_t result, const char* operation) {
            if (result != cudaSuccess) {
                throw std::runtime_error(
                    std::string(operation) + ": " + cudaGetErrorString(result)
                );
            }
        }

        struct SharedBody {
            double x;
            double y;
            double z;
            double mass;
        };

        __global__ void acceleration_tiled_kernel(
            const Body* bodies,
            Acceleration* accelerations,
            std::size_t count,
            double epsilon_squared
        ) {
            extern __shared__ SharedBody shared_bodies[];
            const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

            const bool active = i < count;

            double target_x=0.0, target_y=0.0, target_z=0.0;

            if (active) {
                target_x = bodies[i].x;
                target_y = bodies[i].y;
                target_z = bodies[i].z;
            }

            double ax=0.0, ay=0.0, az=0.0;

            for (std::size_t tile_start = 0; tile_start < count; tile_start += blockDim.x) {
                const std::size_t source_index = tile_start + threadIdx.x;

                if (source_index < count) {
                    shared_bodies[threadIdx.x].x =  bodies[source_index].x;
                    shared_bodies[threadIdx.x].y =  bodies[source_index].y;
                    shared_bodies[threadIdx.x].z =  bodies[source_index].z;
                    shared_bodies[threadIdx.x].mass = bodies[source_index].mass;
                }
                __syncthreads();
                
                // force calculation
                const std::size_t remaining = count - tile_start;

                const std::size_t tile_count = (remaining < blockDim.x) ? remaining : blockDim.x;

                if (active) {
                    for (std::size_t local_index = 0; local_index < tile_count; local_index++) {
                        const std::size_t global_source = tile_start + local_index;
                        if (global_source != i) {
                            const SharedBody source_body = shared_bodies[local_index];

                            // get displacement
                            const double dx = source_body.x - target_x;
                            const double dy = source_body.y - target_y;
                            const double dz = source_body.z - target_z; 

                            // get squared softened distance
                            const double squared_softened_distance = dx*dx + dy*dy + dz*dz + epsilon_squared;

                            // get inverse distance
                            const double inverse_distance = 1.0 / sqrt(squared_softened_distance);

                            // get inverse distance cubed
                            const double inverse_distance_cubed = inverse_distance * inverse_distance * inverse_distance;
                            
                            // get acceleration
                            const double interaction_strength = source_body.mass * inverse_distance_cubed;
                            ax += interaction_strength * dx;
                            ay += interaction_strength * dy;
                            az += interaction_strength * dz;
                        } 
                    }
                }

                __syncthreads();
            }
            // store acceleration
            if (active) {
                accelerations[i] = Acceleration{ax, ay, az};
            }
        }
    }

    Accelerations compute_acceleration_cuda_tiled(
        const Bodies& bodies,
        double epsilon,
        int block_size
    ) {
        Accelerations acceleration_vector(bodies.size());

        if (bodies.empty()) {
            return acceleration_vector;
        }

        if (block_size <= 0) {
            throw std::invalid_argument("block size must be positive");
        }

        Body* in_bodies = nullptr;
        Acceleration* out_acc = nullptr;

        // get bytes needed for vectors
        const std::size_t bodies_bytes = bodies.size() * sizeof(Body);
        const std::size_t acc_bytes = acceleration_vector.size() * sizeof(Acceleration);

        const std::size_t threads_per_block = static_cast<std::size_t>(block_size);
        const std::size_t grid_size = (bodies.size() + threads_per_block - 1) / threads_per_block;
        const std::size_t dynamic_shared_memory_bytes = threads_per_block * sizeof(SharedBody);

        try{
            // allocates memory in GPU
            const cudaError_t b_result = cudaMalloc(reinterpret_cast<void**>(&in_bodies), bodies_bytes);
            check_cuda(b_result, "CUDA Malloc for bodies");

            const cudaError_t a_result = cudaMalloc(reinterpret_cast<void**>(&out_acc), acc_bytes);
            check_cuda(a_result, "CUDA Malloc for acceleration");

            // copies bodies from CPU to allocated GPU
            const cudaError_t copy_b_result = cudaMemcpy(in_bodies, bodies.data(), bodies_bytes, cudaMemcpyHostToDevice);
            check_cuda(copy_b_result, "CUDA Mem copy for bodies");

            // calls kernel for acceleration computing
            acceleration_tiled_kernel<<<grid_size, threads_per_block, dynamic_shared_memory_bytes>>>(in_bodies, out_acc, bodies.size(), epsilon*epsilon);
            const cudaError_t kernel_result = cudaGetLastError();
            check_cuda(kernel_result, "acceleration kernel launch");
            const cudaError_t sync_result = cudaDeviceSynchronize(); // wait for compute device to finish 
            check_cuda(sync_result, "acceleration kernel execution");

            // copy computed acceleration back to CPU
            const cudaError_t copy_a_result = cudaMemcpy(acceleration_vector.data(), out_acc, acc_bytes, cudaMemcpyDeviceToHost);
            check_cuda(copy_a_result, "CUDA to CPU mem copy for acceleration");
        } catch (...) {
            // if error, free bodies and acceleration allocations
            if (in_bodies != nullptr) {
                (void)cudaFree(in_bodies);
            }
            if (out_acc != nullptr) {
                (void)cudaFree(out_acc);
            }
            throw;
        }

        // free CUDA allocation
        const cudaError_t free_body = cudaFree(in_bodies);
        const cudaError_t free_acc = cudaFree(out_acc);
        check_cuda(free_body, "cudaFree bodies"); // only check after both alloc are freed
        check_cuda(free_acc, "cudaFree acceleration");

        return acceleration_vector;
    }

    Bodies simulate_cuda_tiled(
        const Bodies& initial_bodies,
        std::size_t steps,
        double dt,
        double epsilon,
        int block_size,
        CudaTiming* timing
    ) {
        // initialize timing
        if (timing != nullptr) {
            *timing = CudaTiming{};
        }
        
        // copy initial bodies
        Bodies final_bodies = initial_bodies;

        // check that bodies is not empty
        if (final_bodies.empty() || steps == 0) {
            return final_bodies;
        }

        // check all params are valid
        if (dt <= 0.0) {
            throw std::invalid_argument("dt must be positive");
        }

        if (epsilon < 0.0) {
            throw std::invalid_argument("epsilon cannot be negative");
        }

        if (block_size <= 0) {
            throw std::invalid_argument("block size must be positive");
        }
        
        // make 2 local pointers
        Body* device_bodies = nullptr;
        Acceleration* device_accelerations = nullptr;

        // calculate body and acceleration bytes
        const std::size_t body_bytes = final_bodies.size() * sizeof(Body);
        const std::size_t acceleration_bytes = final_bodies.size() * sizeof(Acceleration);

        // calculate threads per block and grid size
        const std::size_t threads_per_block = static_cast<std::size_t>(block_size);
        const std::size_t dynamic_shared_memory_bytes = threads_per_block * sizeof(SharedBody);
        const std::size_t grid_size = (final_bodies.size() + threads_per_block - 1) / threads_per_block;

        const double epsilon_squared = epsilon * epsilon;

        try
        {
            // malloc for bodies and acceleration
            const cudaError_t body_alloc_res = cudaMalloc(reinterpret_cast<void**>(&device_bodies), body_bytes);
            check_cuda(body_alloc_res, "CUDA Malloc for bodies");
            const cudaError_t acc_alloc_res = cudaMalloc(reinterpret_cast<void**>(&device_accelerations), acceleration_bytes);
            check_cuda(acc_alloc_res, "CUDA Malloc for acceleration");

            // copy final_bodies to device_bodies
            const cudaError_t body_copy_res = cudaMemcpy(device_bodies, final_bodies.data(), body_bytes, cudaMemcpyHostToDevice);
            check_cuda(body_copy_res, "CUDA Mem copy for bodies, host to device");

            // run acceleration kernel
            acceleration_tiled_kernel<<<grid_size, threads_per_block, dynamic_shared_memory_bytes>>>(device_bodies, device_accelerations, final_bodies.size(), epsilon_squared);
            const cudaError_t acc_kernel_res = cudaGetLastError();
            check_cuda(acc_kernel_res, "initial acceleration kernel launch");

            // velocity verlet loop
            for (std::size_t step = 0; step < steps; ++step) {
                cuda_detail::kick_drift_kernel<<<grid_size, threads_per_block>>>(device_bodies, device_accelerations, final_bodies.size(), dt);
                const cudaError_t kick_drift_result = cudaGetLastError();
                check_cuda(kick_drift_result, "kick drift kernel launch");

                acceleration_tiled_kernel<<<grid_size, threads_per_block, dynamic_shared_memory_bytes>>>(device_bodies, device_accelerations, final_bodies.size(), epsilon_squared);
                const cudaError_t step_acceleration_result = cudaGetLastError();
                check_cuda(step_acceleration_result, "acceleration kernel launch");

                cuda_detail::final_kick_kernel<<<grid_size, threads_per_block>>>(device_bodies, device_accelerations, final_bodies.size(), dt);
                const cudaError_t final_kick_result = cudaGetLastError();
                check_cuda(final_kick_result, "final kick kernel launch");

            }

            const cudaError_t sync_result = cudaDeviceSynchronize();
            check_cuda(sync_result, "tiled simulation execution");

            const cudaError_t copy_body_dev_to_host = cudaMemcpy(final_bodies.data(), device_bodies, body_bytes, cudaMemcpyDeviceToHost);
            check_cuda(copy_body_dev_to_host, "copy final bodies from device to host");
        }
        catch(...)
        {
            if (device_bodies != nullptr) {
                (void)cudaFree(device_bodies);
            }
            if (device_accelerations != nullptr) {
                (void)cudaFree(device_accelerations);
            }

            throw;
        }

        const cudaError_t free_bodies_result = cudaFree(device_bodies);
        const cudaError_t free_accelerations_result = cudaFree(device_accelerations);
        check_cuda(free_bodies_result, "cudaFree simulation bodies"); // only check after both alloc are freed
        check_cuda(free_accelerations_result, "cudaFree simulation acceleration");

        return final_bodies;
    }

} // end of namespace