#include <ATen/ATen.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cstdint>
#include <vector>

namespace {

constexpr int maximum_stack_size = 256;

template <typename scalar_t, bool write_output>
__device__ std::int64_t traverse(
    std::int64_t target,
    const scalar_t* positions,
    const scalar_t* body_masses,
    const std::int32_t* children,
    const std::int64_t* leaf_offsets,
    const std::int32_t* leaf_counts,
    const std::int64_t* leaf_bodies,
    const scalar_t* centers,
    const scalar_t* half_sizes,
    const scalar_t* node_masses,
    const scalar_t* centers_of_mass,
    const scalar_t* rms_radii,
    scalar_t theta,
    std::int64_t output_offset,
    std::int64_t* output_targets,
    scalar_t* output_positions,
    scalar_t* output_masses,
    scalar_t* output_sizes,
    scalar_t* output_occupancies,
    scalar_t* output_rms_radii,
    bool* output_aggregates,
    std::int32_t* overflow) {
    std::int32_t node_stack[maximum_stack_size];
    bool contains_stack[maximum_stack_size];
    int stack_size = 1;
    node_stack[0] = 0;
    contains_stack[0] = true;
    std::int64_t count = 0;
    const scalar_t* target_position = positions + 3 * target;

    while (stack_size > 0) {
        --stack_size;
        const std::int32_t node = node_stack[stack_size];
        const bool contains_target = contains_stack[stack_size];
        const std::int32_t leaf_count = leaf_counts[node];
        if (leaf_count > 0) {
            const std::int64_t leaf_offset = leaf_offsets[node];
            for (std::int32_t body = 0; body < leaf_count; ++body) {
                const std::int64_t source = leaf_bodies[leaf_offset + body];
                if (source == target) {
                    continue;
                }
                if constexpr (write_output) {
                    const std::int64_t destination = output_offset + count;
                    output_targets[destination] = target;
                    for (int axis = 0; axis < 3; ++axis) {
                        output_positions[3 * destination + axis] =
                            positions[3 * source + axis];
                    }
                    output_masses[destination] = body_masses[source];
                    output_sizes[destination] = scalar_t{0};
                    output_occupancies[destination] = scalar_t{1};
                    output_rms_radii[destination] = scalar_t{0};
                    output_aggregates[destination] = false;
                }
                ++count;
            }
            continue;
        }

        scalar_t squared_distance = scalar_t{0};
        for (int axis = 0; axis < 3; ++axis) {
            const scalar_t difference =
                centers_of_mass[3 * node + axis] - target_position[axis];
            squared_distance += difference * difference;
        }
        const scalar_t distance = sqrt(squared_distance);
        const scalar_t node_size = scalar_t{2} * half_sizes[node];
        if (!contains_target && distance > scalar_t{0} &&
            node_size / distance < theta) {
            if constexpr (write_output) {
                const std::int64_t destination = output_offset + count;
                output_targets[destination] = target;
                for (int axis = 0; axis < 3; ++axis) {
                    output_positions[3 * destination + axis] =
                        centers_of_mass[3 * node + axis];
                }
                output_masses[destination] = node_masses[node];
                output_sizes[destination] = node_size;
                output_occupancies[destination] =
                    static_cast<scalar_t>(leaf_offsets[node]);
                output_rms_radii[destination] = rms_radii[node];
                output_aggregates[destination] = true;
            }
            ++count;
            continue;
        }

        int target_octant = 0;
        target_octant |= static_cast<int>(target_position[0] >= centers[3 * node]);
        target_octant |= static_cast<int>(target_position[1] >= centers[3 * node + 1]) << 1;
        target_octant |= static_cast<int>(target_position[2] >= centers[3 * node + 2]) << 2;
        for (int child_octant = 0; child_octant < 8; ++child_octant) {
            const std::int32_t child = children[8 * node + child_octant];
            if (child < 0) {
                continue;
            }
            if (stack_size >= maximum_stack_size) {
                atomicExch(overflow, 1);
                return count;
            }
            node_stack[stack_size] = child;
            contains_stack[stack_size] =
                contains_target && child_octant == target_octant;
            ++stack_size;
        }
    }
    return count;
}

template <typename scalar_t>
__global__ void count_kernel(
    std::int64_t count,
    const scalar_t* positions,
    const scalar_t* body_masses,
    const std::int32_t* children,
    const std::int64_t* leaf_offsets,
    const std::int32_t* leaf_counts,
    const std::int64_t* leaf_bodies,
    const scalar_t* centers,
    const scalar_t* half_sizes,
    const scalar_t* node_masses,
    const scalar_t* centers_of_mass,
    const scalar_t* rms_radii,
    scalar_t theta,
    std::int64_t* counts,
    std::int32_t* overflow) {
    const std::int64_t target = blockIdx.x * blockDim.x + threadIdx.x;
    if (target >= count) {
        return;
    }
    counts[target] = traverse<scalar_t, false>(
        target, positions, body_masses, children, leaf_offsets, leaf_counts, leaf_bodies,
        centers, half_sizes, node_masses, centers_of_mass, rms_radii, theta, 0,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, overflow);
}

template <typename scalar_t>
__global__ void fill_kernel(
    std::int64_t count,
    const scalar_t* positions,
    const scalar_t* body_masses,
    const std::int32_t* children,
    const std::int64_t* leaf_offsets,
    const std::int32_t* leaf_counts,
    const std::int64_t* leaf_bodies,
    const scalar_t* centers,
    const scalar_t* half_sizes,
    const scalar_t* node_masses,
    const scalar_t* centers_of_mass,
    const scalar_t* rms_radii,
    scalar_t theta,
    const std::int64_t* offsets,
    std::int64_t* output_targets,
    scalar_t* output_positions,
    scalar_t* output_masses,
    scalar_t* output_sizes,
    scalar_t* output_occupancies,
    scalar_t* output_rms_radii,
    bool* output_aggregates,
    std::int32_t* overflow) {
    const std::int64_t target = blockIdx.x * blockDim.x + threadIdx.x;
    if (target >= count) {
        return;
    }
    traverse<scalar_t, true>(
        target, positions, body_masses, children, leaf_offsets, leaf_counts, leaf_bodies,
        centers, half_sizes, node_masses, centers_of_mass, rms_radii, theta,
        offsets[target], output_targets, output_positions, output_masses,
        output_sizes, output_occupancies, output_rms_radii, output_aggregates,
        overflow);

}

}  // namespace

std::vector<at::Tensor> launch_hierarchical_cuda(
    at::Tensor positions,
    at::Tensor masses,
    at::Tensor children,
    at::Tensor leaf_offsets,
    at::Tensor leaf_counts,
    at::Tensor leaf_bodies,
    at::Tensor centers,
    at::Tensor half_sizes,
    at::Tensor node_masses,
    at::Tensor centers_of_mass,
    at::Tensor rms_radii,
    double theta) {
    const auto count = positions.size(0);
    auto long_options = positions.options().dtype(at::kLong);
    auto int_options = positions.options().dtype(at::kInt);
    auto counts = at::zeros({count}, long_options);
    auto overflow = at::zeros({1}, int_options);
    constexpr int threads = 128;
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    const auto stream = at::cuda::getDefaultCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(positions.scalar_type(), "hierarchical_count", [&] {
        count_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            count, positions.data_ptr<scalar_t>(), masses.data_ptr<scalar_t>(),
            children.data_ptr<std::int32_t>(),
            leaf_offsets.data_ptr<std::int64_t>(), leaf_counts.data_ptr<std::int32_t>(),
            leaf_bodies.data_ptr<std::int64_t>(), centers.data_ptr<scalar_t>(),
            half_sizes.data_ptr<scalar_t>(), node_masses.data_ptr<scalar_t>(),
            centers_of_mass.data_ptr<scalar_t>(), rms_radii.data_ptr<scalar_t>(),
            static_cast<scalar_t>(theta), counts.data_ptr<std::int64_t>(),
            overflow.data_ptr<std::int32_t>());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto offsets = at::zeros({count + 1}, long_options);
    offsets.slice(0, 1).copy_(counts.cumsum(0));
    const auto total = offsets.index({count}).item<std::int64_t>();
    auto targets = at::empty({total}, long_options);
    auto source_positions = at::empty({total, 3}, positions.options());
    auto source_masses = at::empty({total}, positions.options());
    auto source_sizes = at::empty({total}, positions.options());
    auto occupancies = at::empty({total}, positions.options());
    auto source_rms = at::empty({total}, positions.options());
    auto aggregates = at::empty({total}, positions.options().dtype(at::kBool));

    AT_DISPATCH_FLOATING_TYPES(positions.scalar_type(), "hierarchical_fill", [&] {
        fill_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            count, positions.data_ptr<scalar_t>(), masses.data_ptr<scalar_t>(),
            children.data_ptr<std::int32_t>(), leaf_offsets.data_ptr<std::int64_t>(),
            leaf_counts.data_ptr<std::int32_t>(), leaf_bodies.data_ptr<std::int64_t>(),
            centers.data_ptr<scalar_t>(), half_sizes.data_ptr<scalar_t>(),
            node_masses.data_ptr<scalar_t>(), centers_of_mass.data_ptr<scalar_t>(),
            rms_radii.data_ptr<scalar_t>(), static_cast<scalar_t>(theta),
            offsets.data_ptr<std::int64_t>(), targets.data_ptr<std::int64_t>(),
            source_positions.data_ptr<scalar_t>(), source_masses.data_ptr<scalar_t>(),
            source_sizes.data_ptr<scalar_t>(), occupancies.data_ptr<scalar_t>(),
            source_rms.data_ptr<scalar_t>(), aggregates.data_ptr<bool>(),
            overflow.data_ptr<std::int32_t>());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    TORCH_CHECK(overflow.cpu().item<std::int32_t>() == 0, "CUDA octree traversal stack overflow");
    return {targets, source_positions, source_masses, source_sizes, occupancies,
            source_rms, aggregates};
}
