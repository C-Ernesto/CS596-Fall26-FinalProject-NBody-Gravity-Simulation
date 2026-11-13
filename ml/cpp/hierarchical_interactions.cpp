#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

struct Node {
    std::array<double, 3> center{};
    double half_size = 0.0;
    std::vector<std::int64_t> indices;
    double mass = 0.0;
    std::array<double, 3> center_of_mass{};
    double rms_radius = 0.0;
    std::array<std::unique_ptr<Node>, 8> children;

    bool is_leaf() const {
        return std::all_of(children.begin(), children.end(),
                           [](const auto& child) { return child == nullptr; });
    }
};

struct Interaction {
    std::int64_t target = 0;
    std::array<double, 3> source_position{};
    double source_mass = 0.0;
    double source_size = 0.0;
    double source_occupancy = 0.0;
    double source_rms_radius = 0.0;
    bool aggregate = false;
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
    if (!(node->mass > 0.0)) {
        throw std::runtime_error("all octree nodes must have positive mass");
    }
    for (double& value : node->center_of_mass) {
        value /= node->mass;
    }
    double weighted_squared_radius = 0.0;
    for (const auto index : node->indices) {
        double squared_radius = 0.0;
        for (int axis = 0; axis < 3; ++axis) {
            const double difference =
                positions[3 * index + axis] - node->center_of_mass[axis];
            squared_radius += difference * difference;
        }
        weighted_squared_radius += masses[index] * squared_radius;
    }
    node->rms_radius = std::sqrt(weighted_squared_radius / node->mass);

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
            const double sign = (child_octant & (1 << axis)) ? 1.0 : -1.0;
            child_center[axis] = center[axis] + sign * child_half;
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

void traverse_target(
    const Node& root,
    const double* positions,
    const double* masses,
    std::int64_t target,
    double theta,
    std::vector<Interaction>& output) {
    struct StackEntry {
        const Node* node;
        bool contains_target;
    };
    std::vector<StackEntry> stack;
    stack.push_back({&root, true});
    const double* target_position = positions + 3 * target;

    while (!stack.empty()) {
        const StackEntry entry = stack.back();
        stack.pop_back();
        const Node& node = *entry.node;
        if (node.is_leaf()) {
            for (const auto source : node.indices) {
                if (source == target) {
                    continue;
                }
                Interaction interaction;
                interaction.target = target;
                for (int axis = 0; axis < 3; ++axis) {
                    interaction.source_position[axis] = positions[3 * source + axis];
                }
                interaction.source_mass = masses[source];
                interaction.source_occupancy = 1.0;
                output.push_back(interaction);
            }
            continue;
        }

        double squared_distance = 0.0;
        for (int axis = 0; axis < 3; ++axis) {
            const double difference = node.center_of_mass[axis] - target_position[axis];
            squared_distance += difference * difference;
        }
        const double distance = std::sqrt(squared_distance);
        const double node_size = 2.0 * node.half_size;
        if (!entry.contains_target && distance > 0.0 && node_size / distance < theta) {
            Interaction interaction;
            interaction.target = target;
            interaction.source_position = node.center_of_mass;
            interaction.source_mass = node.mass;
            interaction.source_size = node_size;
            interaction.source_occupancy = static_cast<double>(node.indices.size());
            interaction.source_rms_radius = node.rms_radius;
            interaction.aggregate = true;
            output.push_back(interaction);
            continue;
        }

        const int target_octant = octant(target_position, node.center);
        for (int child_octant = 0; child_octant < 8; ++child_octant) {
            if (node.children[child_octant]) {
                stack.push_back(
                    {node.children[child_octant].get(),
                     entry.contains_target && child_octant == target_octant});
            }
        }
    }
}

PyObject* build_interactions(PyObject*, PyObject* arguments) {
    PyObject* positions_object = nullptr;
    PyObject* masses_object = nullptr;
    double theta = 0.0;
    int leaf_capacity = 0;
    int maximum_depth = 0;
    if (!PyArg_ParseTuple(arguments, "OOdii", &positions_object, &masses_object,
                          &theta, &leaf_capacity, &maximum_depth)) {
        return nullptr;
    }
    auto* positions_array = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(positions_object, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    auto* masses_array = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(masses_object, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    if (!positions_array || !masses_array) {
        Py_XDECREF(positions_array);
        Py_XDECREF(masses_array);
        return nullptr;
    }

    PyObject* result = nullptr;
    try {
        if (theta < 0.0 || leaf_capacity <= 0 || maximum_depth <= 0) {
            throw std::invalid_argument("invalid hierarchy configuration");
        }
        if (PyArray_NDIM(positions_array) != 2 ||
            PyArray_DIM(positions_array, 1) != 3) {
            throw std::invalid_argument("positions must have shape [N, 3]");
        }
        const auto count = static_cast<std::int64_t>(PyArray_DIM(positions_array, 0));
        if (count <= 0 || PyArray_SIZE(masses_array) != count) {
            throw std::invalid_argument("masses must contain one value per position");
        }
        const auto* positions = static_cast<const double*>(PyArray_DATA(positions_array));
        const auto* masses = static_cast<const double*>(PyArray_DATA(masses_array));
        auto root = build_tree(positions, masses, count, leaf_capacity, maximum_depth);
        std::vector<std::vector<Interaction>> per_target(static_cast<std::size_t>(count));

        Py_BEGIN_ALLOW_THREADS
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (std::int64_t target = 0; target < count; ++target) {
            traverse_target(*root, positions, masses, target, theta,
                            per_target[static_cast<std::size_t>(target)]);
        }
        Py_END_ALLOW_THREADS

        std::size_t interaction_count = 0;
        for (const auto& interactions : per_target) {
            interaction_count += interactions.size();
        }
        npy_intp vector_shape[1] = {static_cast<npy_intp>(interaction_count)};
        npy_intp position_shape[2] = {static_cast<npy_intp>(interaction_count), 3};
        auto* targets = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, vector_shape, NPY_INT64));
        auto* sources = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(2, position_shape, NPY_DOUBLE));
        auto* source_masses = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, vector_shape, NPY_DOUBLE));
        auto* source_sizes = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, vector_shape, NPY_DOUBLE));
        auto* occupancies = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, vector_shape, NPY_DOUBLE));
        auto* rms_radii = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, vector_shape, NPY_DOUBLE));
        auto* aggregates = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, vector_shape, NPY_BOOL));
        if (!targets || !sources || !source_masses || !source_sizes || !occupancies ||
            !rms_radii || !aggregates) {
            throw std::bad_alloc();
        }

        auto* target_data = static_cast<std::int64_t*>(PyArray_DATA(targets));
        auto* source_data = static_cast<double*>(PyArray_DATA(sources));
        auto* mass_data = static_cast<double*>(PyArray_DATA(source_masses));
        auto* size_data = static_cast<double*>(PyArray_DATA(source_sizes));
        auto* occupancy_data = static_cast<double*>(PyArray_DATA(occupancies));
        auto* rms_data = static_cast<double*>(PyArray_DATA(rms_radii));
        auto* aggregate_data = static_cast<npy_bool*>(PyArray_DATA(aggregates));
        std::size_t destination = 0;
        for (const auto& interactions : per_target) {
            for (const auto& interaction : interactions) {
                target_data[destination] = interaction.target;
                for (int axis = 0; axis < 3; ++axis) {
                    source_data[3 * destination + axis] = interaction.source_position[axis];
                }
                mass_data[destination] = interaction.source_mass;
                size_data[destination] = interaction.source_size;
                occupancy_data[destination] = interaction.source_occupancy;
                rms_data[destination] = interaction.source_rms_radius;
                aggregate_data[destination] = interaction.aggregate ? 1 : 0;
                ++destination;
            }
        }

        result = PyDict_New();
        PyDict_SetItemString(result, "target_indices", reinterpret_cast<PyObject*>(targets));
        PyDict_SetItemString(result, "source_positions", reinterpret_cast<PyObject*>(sources));
        PyDict_SetItemString(result, "source_masses", reinterpret_cast<PyObject*>(source_masses));
        PyDict_SetItemString(result, "source_sizes", reinterpret_cast<PyObject*>(source_sizes));
        PyDict_SetItemString(result, "source_occupancies", reinterpret_cast<PyObject*>(occupancies));
        PyDict_SetItemString(result, "source_rms_radii", reinterpret_cast<PyObject*>(rms_radii));
        PyDict_SetItemString(result, "is_aggregate", reinterpret_cast<PyObject*>(aggregates));
        Py_DECREF(targets);
        Py_DECREF(sources);
        Py_DECREF(source_masses);
        Py_DECREF(source_sizes);
        Py_DECREF(occupancies);
        Py_DECREF(rms_radii);
        Py_DECREF(aggregates);
    } catch (const std::exception& error) {
        PyErr_SetString(PyExc_ValueError, error.what());
    }
    Py_DECREF(positions_array);
    Py_DECREF(masses_array);
    return result;
}

PyObject* max_threads(PyObject*, PyObject*) {
#ifdef _OPENMP
    return PyLong_FromLong(omp_get_max_threads());
#else
    return PyLong_FromLong(1);
#endif
}

PyObject* set_num_threads(PyObject*, PyObject* arguments) {
    int threads = 0;
    if (!PyArg_ParseTuple(arguments, "i", &threads)) {
        return nullptr;
    }
    if (threads <= 0) {
        PyErr_SetString(PyExc_ValueError, "thread count must be positive");
        return nullptr;
    }
#ifdef _OPENMP
    omp_set_num_threads(threads);
#endif
    Py_RETURN_NONE;
}

PyMethodDef methods[] = {
    {"build_interactions", build_interactions, METH_VARARGS,
     "Build exact-near/aggregate-far interactions."},
    {"max_threads", max_threads, METH_NOARGS,
     "Return the maximum OpenMP thread count."},
    {"set_num_threads", set_num_threads, METH_VARARGS,
     "Set the OpenMP thread count used by traversal."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_hierarchical_cpp",
    "Compiled hierarchical interaction builder.",
    -1,
    methods,
};

}  // namespace

PyMODINIT_FUNC PyInit__hierarchical_cpp() {
    import_array();
    return PyModule_Create(&module);
}
