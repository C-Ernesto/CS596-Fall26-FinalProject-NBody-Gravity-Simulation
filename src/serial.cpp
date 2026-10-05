#include "nbody/simulation.hpp"

#include <stdexcept>

namespace nbody {

Accelerations compute_acceleration_serial(const Bodies&, double) {
    throw std::logic_error("serial acceleration is planned for milestone 2");
}

}  // namespace nbody

