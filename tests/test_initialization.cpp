#include "nbody/simulation.hpp"

#include <iostream>

int main() {
    const nbody::Bodies first = nbody::initialize_bodies(8, 12345);
    const nbody::Bodies second = nbody::initialize_bodies(8, 12345);

    if (first.size() != second.size()) return 1;
    for (std::size_t i = 0; i < first.size(); ++i) {
        if (first[i].x != second[i].x || first[i].y != second[i].y ||
            first[i].z != second[i].z || first[i].vx != second[i].vx ||
            first[i].vy != second[i].vy || first[i].vz != second[i].vz ||
            first[i].mass != second[i].mass) {
            std::cerr << "initialization differs at body " << i << '\n';
            return 1;
        }
    }
    return 0;
}

