#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "navigation.hpp"

class PressureSolver {

    float tuningConstant = 0.97;
    float tolerance = 0.001;
    int maxit = 1000;
    Eigen::Vector3i gridSize;
    Eigen::Vector3f c;

    std::vector<float> preconditioner;
    Eigen::VectorXf auxiliaryVec;


    inline int cellIndex(int i, int j, int k) const {
        return i + gridSize.x() * (j + gridSize.y() * k);
    }
    void applyA(const Eigen::VectorXf& v, Eigen::VectorXf& Av);
    void applyPreconditioner(const Eigen::VectorXf& v, Eigen::VectorXf& Mv);
    void buildPreconditioner();

    float getA(Index3D idx, Axis axis);
    float getDiagA(Index3D idx);
public:
    void initialize(Eigen::Vector3i _gridSize) {
        gridSize = _gridSize;
        c.x() = float(gridSize.x() * gridSize.x());
        c.z() = float(gridSize.z() * gridSize.z());
        c.y() = float(gridSize.y() * gridSize.y());
        auxiliaryVec = Eigen::VectorXf::Zero(gridSize.x() * gridSize.y() * gridSize.z());
        buildPreconditioner();
    }
    
    Eigen::VectorXf computePressure(const Eigen::VectorXf& divergence);

    float getTol() const { return tolerance; }
};