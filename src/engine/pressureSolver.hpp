#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>

class PressureSolver {

    float tolerance = 0.1;
    int maxit = 1000;

public:
    Eigen::VectorXf computePressure(const Eigen::VectorXf& divergence, const Eigen::SparseMatrix<float>& A, const Eigen::IncompleteCholesky<float>& ichol);

    float getTol() const { return tolerance; }
};