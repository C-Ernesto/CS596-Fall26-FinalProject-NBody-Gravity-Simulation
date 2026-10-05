#pragma once

namespace nbody {

struct Body {
    double x{};
    double y{};
    double z{};
    double vx{};
    double vy{};
    double vz{};
    double mass{1.0};
};

struct Acceleration {
    double x{};
    double y{};
    double z{};
};

}  // namespace nbody

