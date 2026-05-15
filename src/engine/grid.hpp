#pragma once

#include <Eigen/Dense>
#include <vector>

class Grid {
    Eigen::MatrixXd velocityX;
    Eigen::MatrixXd velocityY;
    Eigen::MatrixXd pressure;

    float rho = 1000;
    float timestep = 0.001;

    Eigen::Vector2d gridSize;
public:
    void initialize(Eigen::Vector2d _gridSize);

    const Eigen::MatrixXd& getPressureValues() const { return pressure; }
    const Eigen::MatrixXd& getVelocityXValues() const { return velocityX; }
    const Eigen::MatrixXd& getVelocityYValues() const { return velocityY; }

    Eigen::Vector2d getDimensions() const { return gridSize; }
};