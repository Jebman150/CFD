#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "navigation.hpp"
#include "grid.hpp"

namespace engine {

using namespace navigation;

class PressureSolver {

    float tuningConstant = 0.97;
    float tolerance = 0.0001;
    int maxit = 100;
    const Grid* grid;
    Eigen::Vector3i gridSize;
    Eigen::Vector3f c;

    std::vector<float> preconditioner;
    Eigen::VectorXf auxiliaryVec;


    inline int cellIndex(int i, int j, int k) const {
        return i + gridSize.x() * (j + gridSize.y() * k);
    }
    inline int cellIndex(Index3D idx) const {
        return idx.i + gridSize.x() * (idx.j + gridSize.y() * idx.k);
    }
    void applyA(const Eigen::VectorXf& v, Eigen::VectorXf& Av);
    void applyPreconditioner(const Eigen::VectorXf& v, Eigen::VectorXf& Mv);
    void buildPreconditioner();

    float getA(Index3D idx, Axis axis);
    float getDiagA(Index3D idx);
public:
    void initialize(const Grid& _grid, Eigen::Vector3f _d) {
        grid = &_grid;
        gridSize = {grid->getWidth(), grid->getHeight(), grid->getDepth()};
        c.x() = 1.f/(_d.x() * _d.x());
        c.y() = 1.f/(_d.y() * _d.y());
        c.z() = 1.f/(_d.z() * _d.z());
        auxiliaryVec = Eigen::VectorXf::Zero(gridSize.x() * gridSize.y() * gridSize.z());
        buildPreconditioner();
    }
    
    Eigen::VectorXf computePressure(const Eigen::VectorXf& divergence);

    float getTol() const { return tolerance; }
};

}