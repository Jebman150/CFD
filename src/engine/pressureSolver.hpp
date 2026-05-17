#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>

class PressureSolver {

    float tuningConstant = 0.97;
    float tolerance = 0.0001;
    int maxit = 1000;

public:
    Eigen::VectorXd computePressure(const Eigen::VectorXd& divergence, const Eigen::SparseMatrix<double>& A);
};