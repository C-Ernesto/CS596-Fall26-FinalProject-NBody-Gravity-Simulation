// Milestone placeholder: one CUDA thread will compute acceleration for one body.
// Keep simulation state on the GPU across integration steps.

#include <cstddef>
#include <cmath>
#include "nbody/simulation.hpp"
#include <cuda_runtime.h>
#include <string>
#include <stdexcept>

namespace nbody {

    namespace {
        // helper function for CUDA error-checking
        void check_cuda(cudaError_t result, const char* operation) {
            if (result != cudaSuccess) {
                throw std::runtime_error(
                    std::string(operation) + ": " + cudaGetErrorString(result)
                );
            }
        }
    }

    __global__ void acceleration_kernel(
        const Body* bodies,
        Acceleration* accelerations,
        std::size_t count, 
        double epsilon_squared
    ) {
        // get global thread index
        const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;

        if (i >= count) return;

        const Body current_body = bodies[i];
        double ax=0.0, ay=0.0, az=0.0;

        for (std::size_t j=0; j<count; j++){
            //skip i=j
            if (i != j){
                const Body source_body = bodies[j];
                
                // get displacement
                const double dx = source_body.x - current_body.x;
                const double dy = source_body.y - current_body.y;
                const double dz = source_body.z - current_body.z; 

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

        // store acceleration
        accelerations[i] = Acceleration{ax, ay, az};
    }

    Accelerations compute_acceleration_cuda_naive(
        const Bodies& bodies,
        double epsilon,
        int block_size
    ) {
        Accelerations acceleration_vector(bodies.size());

        Body* in_bodies = nullptr;
        Acceleration* out_acc = nullptr;

        // get bytes needed for vectors
        const std::size_t bodies_bytes = bodies.size() * sizeof(Body);
        const std::size_t acc_bytes = acceleration_vector.size() * sizeof(Acceleration);

        if (bodies.empty()) {
            return acceleration_vector;
        }

        if (block_size <= 0) {
            throw std::invalid_argument("block size must be positive");
        }

        // how many cuda blocks are required to store all bodies
        const std::size_t threads_per_block = static_cast<std::size_t>(block_size);
        const std::size_t grid_size = (bodies.size() + threads_per_block - 1) / threads_per_block;

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
            acceleration_kernel<<<grid_size, threads_per_block>>>(in_bodies, out_acc, bodies.size(), epsilon*epsilon);
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
    
}