#include "nbody/simulation.hpp"

#include <stdexcept>

namespace nbody {

Accelerations compute_acceleration_openmp(const Bodies&, double) {
    throw std::logic_error("OpenMP acceleration is planned after serial validation");
}

}  // namespace nbody

