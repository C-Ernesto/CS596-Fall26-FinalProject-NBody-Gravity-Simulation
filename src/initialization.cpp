#include "nbody/simulation.hpp"

#include <random>

namespace nbody {

Bodies initialize_bodies(std::size_t count, std::uint64_t seed) {
    std::mt19937_64 generator(seed);
    std::uniform_real_distribution<double> position(-1.0, 1.0);
    std::uniform_real_distribution<double> velocity(-0.01, 0.01);

    Bodies bodies(count);
    for (Body& body : bodies) {
        body.x = position(generator);
        body.y = position(generator);
        body.z = position(generator);
        body.vx = velocity(generator);
        body.vy = velocity(generator);
        body.vz = velocity(generator);
        body.mass = 1.0 / static_cast<double>(count);
    }
    return bodies;
}

}  // namespace nbody

