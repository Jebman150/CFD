#pragma once

#include <Eigen/Dense>

struct SimParameter {
    const float rho = 1000;
    const float timestep = 0.001;

    const Eigen::Vector2d gridSize = {10, 10};
};