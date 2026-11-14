#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

namespace py = pybind11;
using Clock = std::chrono::steady_clock;

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
    double theta);

namespace {

struct Node {
    std::array<double, 3> center{};
    double half_size = 0.0;
    std::vector<std::int64_t> indices;
    double mass = 0.0;
    std::array<double, 3> center_of_mass{};
    double rms_radius = 0.0;
    std::array<std::unique_ptr<Node>, 8> children;
};

int octant(const double* position, const std::array<double, 3>& center) {
    return static_cast<int>(position[0] >= center[0]) |
           (static_cast<int>(position[1] >= center[1]) << 1) |
           (static_cast<int>(position[2] >= center[2]) << 2);
}

std::unique_ptr<Node> build_node(
    const double* positions,
    const double* masses,
    std::vector<std::int64_t> indices,
    const std::array<double, 3>& center,
    double half_size,
    int depth,
    int leaf_capacity,
    int maximum_depth) {
    auto node = std::make_unique<Node>();
    node->center = center;
    node->half_size = half_size;
    node->indices = std::move(indices);
    for (const auto index : node->indices) {
        const double mass = masses[index];
        node->mass += mass;
        for (int axis = 0; axis < 3; ++axis) {
            node->center_of_mass[axis] += mass * positions[3 * index + axis];
        }
    }
    TORCH_CHECK(node->mass > 0.0, "octree nodes require positive mass");
    for (auto& value : node->center_of_mass) {
        value /= node->mass;
    }
    double weighted_radius = 0.0;
    for (const auto index : node->indices) {
        double squared_radius = 0.0;
        for (int axis = 0; axis < 3; ++axis) {
            const double difference =
                positions[3 * index + axis] - node->center_of_mass[axis];
            squared_radius += difference * difference;
        }
        weighted_radius += masses[index] * squared_radius;
    }
    node->rms_radius = std::sqrt(weighted_radius / node->mass);
    if (static_cast<int>(node->indices.size()) <= leaf_capacity ||
        depth >= maximum_depth) {
        return node;
    }

    std::array<std::vector<std::int64_t>, 8> groups;
    int nonempty = 0;
    for (const auto index : node->indices) {
        auto& group = groups[octant(positions + 3 * index, center)];
        if (group.empty()) {
            ++nonempty;
        }
        group.push_back(index);
    }
    if (nonempty == 1 && half_size <= 1.0e-14) {
        return node;
    }
    const double child_half = 0.5 * half_size;
    for (int child_octant = 0; child_octant < 8; ++child_octant) {
        if (groups[child_octant].empty()) {
            continue;
        }
        std::array<double, 3> child_center{};
        for (int axis = 0; axis < 3; ++axis) {
            child_center[axis] = center[axis] +
                ((child_octant & (1 << axis)) ? child_half : -child_half);
        }
        node->children[child_octant] = build_node(
            positions, masses, std::move(groups[child_octant]), child_center,
            child_half, depth + 1, leaf_capacity, maximum_depth);
    }
    return node;
}

std::unique_ptr<Node> build_tree(
    const double* positions,
    const double* masses,
    std::int64_t count,
    int leaf_capacity,
    int maximum_depth) {
    std::array<double, 3> minimum{positions[0], positions[1], positions[2]};
    auto maximum = minimum;
    for (std::int64_t index = 1; index < count; ++index) {
        for (int axis = 0; axis < 3; ++axis) {
            minimum[axis] = std::min(minimum[axis], positions[3 * index + axis]);
            maximum[axis] = std::max(maximum[axis], positions[3 * index + axis]);
        }
    }
    std::array<double, 3> center{};
    double half_size = 1.0e-12;
    for (int axis = 0; axis < 3; ++axis) {
        center[axis] = 0.5 * (minimum[axis] + maximum[axis]);
        half_size = std::max(half_size, 0.5 * (maximum[axis] - minimum[axis]));
    }
    half_size *= 1.000001;
    std::vector<std::int64_t> indices(static_cast<std::size_t>(count));
    for (std::int64_t index = 0; index < count; ++index) {
        indices[static_cast<std::size_t>(index)] = index;
    }
    return build_node(positions, masses, std::move(indices), center, half_size,
                      0, leaf_capacity, maximum_depth);
}

struct FlatTree {
    std::vector<std::int32_t> children;
    std::vector<std::int64_t> leaf_offsets;
    std::vector<std::int32_t> leaf_counts;
    std::vector<std::int64_t> leaf_bodies;
    std::vector<double> centers;
    std::vector<double> half_sizes;
    std::vector<double> masses;
    std::vector<double> centers_of_mass;
    std::vector<double> rms_radii;
};

std::int32_t flatten(const Node& node, FlatTree& tree) {
    const auto index = static_cast<std::int32_t>(tree.half_sizes.size());
    tree.children.resize(tree.children.size() + 8, -1);
    tree.leaf_offsets.push_back(static_cast<std::int64_t>(node.indices.size()));
    tree.leaf_counts.push_back(0);
    tree.centers.insert(tree.centers.end(), node.center.begin(), node.center.end());
    tree.half_sizes.push_back(node.half_size);
    tree.masses.push_back(node.mass);
    tree.centers_of_mass.insert(
        tree.centers_of_mass.end(), node.center_of_mass.begin(), node.center_of_mass.end());
    tree.rms_radii.push_back(node.rms_radius);

    bool is_leaf = true;
    for (const auto& child : node.children) {
        is_leaf = is_leaf && !child;
    }
    if (is_leaf) {
        tree.leaf_offsets[index] = static_cast<std::int64_t>(tree.leaf_bodies.size());
        tree.leaf_counts[index] = static_cast<std::int32_t>(node.indices.size());
        tree.leaf_bodies.insert(
            tree.leaf_bodies.end(), node.indices.begin(), node.indices.end());
        return index;
    }
    for (int child_octant = 0; child_octant < 8; ++child_octant) {
        if (node.children[child_octant]) {
            tree.children[8 * index + child_octant] =
                flatten(*node.children[child_octant], tree);
        }
    }
    return index;
}

template <typename T>
torch::Tensor vector_tensor(
    std::vector<T>& values,
    const std::vector<std::int64_t>& shape,
    torch::ScalarType dtype,
    const torch::Device& device) {
    auto cpu_options = torch::TensorOptions().dtype(dtype).device(torch::kCPU);
    return torch::from_blob(values.data(), shape, cpu_options).clone().to(device);
}

double seconds_between(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double>(end - start).count();
}

py::dict build_interactions_cuda(
    torch::Tensor positions,
    torch::Tensor masses,
    double theta,
    int leaf_capacity,
    int maximum_depth) {
    TORCH_CHECK(positions.is_cuda(), "CUDA interaction backend requires CUDA positions");
    TORCH_CHECK(masses.is_cuda(), "CUDA interaction backend requires CUDA masses");
    TORCH_CHECK(positions.dim() == 2 && positions.size(1) == 3,
                "positions must have shape [N, 3]");
    TORCH_CHECK(positions.scalar_type() == torch::kFloat32 ||
                positions.scalar_type() == torch::kFloat64,
                "positions must be float32 or float64");
    TORCH_CHECK(theta >= 0.0 && leaf_capacity > 0 && maximum_depth > 0,
                "invalid hierarchy configuration");
    const c10::cuda::CUDAGuard device_guard(positions.device());
    cudaDeviceSynchronize();
    const auto started = Clock::now();
    auto positions_cpu = positions.to(
        torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat64)).contiguous();
    auto masses_cpu = masses.reshape({-1}).to(
        torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat64)).contiguous();
    const auto copied = Clock::now();
    const auto count = positions.size(0);
    TORCH_CHECK(count > 0 && masses_cpu.numel() == count,
                "masses must contain one value per body");
    auto root = build_tree(
        positions_cpu.data_ptr<double>(), masses_cpu.data_ptr<double>(), count,
        leaf_capacity, maximum_depth);
    FlatTree tree;
    flatten(*root, tree);
    const auto built = Clock::now();

    const auto device = positions.device();
    const auto scalar_type = positions.scalar_type();
    const auto nodes = static_cast<std::int64_t>(tree.half_sizes.size());
    auto children = vector_tensor(
        tree.children, {nodes, 8}, torch::kInt32, device);
    auto leaf_offsets = vector_tensor(
        tree.leaf_offsets, {nodes}, torch::kInt64, device);
    auto leaf_counts = vector_tensor(
        tree.leaf_counts, {nodes}, torch::kInt32, device);
    auto leaf_bodies = vector_tensor(
        tree.leaf_bodies,
        {static_cast<std::int64_t>(tree.leaf_bodies.size())},
        torch::kInt64,
        device);
    auto centers = vector_tensor(tree.centers, {nodes, 3}, torch::kFloat64, device)
        .to(scalar_type);
    auto half_sizes = vector_tensor(tree.half_sizes, {nodes}, torch::kFloat64, device)
        .to(scalar_type);
    auto node_masses = vector_tensor(tree.masses, {nodes}, torch::kFloat64, device)
        .to(scalar_type);
    auto centers_of_mass = vector_tensor(
        tree.centers_of_mass, {nodes, 3}, torch::kFloat64, device).to(scalar_type);
    auto rms_radii = vector_tensor(tree.rms_radii, {nodes}, torch::kFloat64, device)
        .to(scalar_type);
    cudaDeviceSynchronize();
    const auto transferred = Clock::now();
    auto outputs = launch_hierarchical_cuda(
        positions.contiguous(), masses.reshape({-1}).contiguous(), children,
        leaf_offsets, leaf_counts, leaf_bodies, centers, half_sizes, node_masses,
        centers_of_mass, rms_radii, theta);
    const auto traversed = Clock::now();
    const auto aggregate_count = outputs[6].sum().item<std::int64_t>();
    const auto total = outputs[0].numel();

    py::dict result;
    result["target_indices"] = outputs[0];
    result["source_positions"] = outputs[1];
    result["source_masses"] = outputs[2];
    result["source_sizes"] = outputs[3];
    result["source_occupancies"] = outputs[4];
    result["source_rms_radii"] = outputs[5];
    result["is_aggregate"] = outputs[6];
    result["body_interactions"] = total - aggregate_count;
    result["aggregate_interactions"] = aggregate_count;
    result["host_copy_seconds"] = seconds_between(started, copied);
    result["tree_seconds"] = seconds_between(copied, built);
    result["interaction_transfer_seconds"] = seconds_between(built, transferred);
    result["gpu_traversal_seconds"] = seconds_between(transferred, traversed);
    return result;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("build_interactions", &build_interactions_cuda,
               "Build hierarchical interactions with CUDA traversal");
}
