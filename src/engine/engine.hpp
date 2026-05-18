#pragma once

#include "grid.hpp"
#include "pressureSolver.hpp"

class Engine {
    Grid grid;
    Eigen::ConjugateGradient<
        Eigen::SparseMatrix<float>,
        Eigen::Lower|Eigen::Upper,
        Eigen::IncompleteCholesky<float>
    > pressureSolver;

    float deltaT;

    float cTarget = 0.8;
    float density = 1000.f;
    float currentT = 0;
    Eigen::SparseMatrix<float> laplacian;

public:
    void initSim();
    void adjustTimestep();
    void checkCFDCondition();
    void applyBoundaryCondition();
    void advect();
    void advectVelocities();
    void advectScalarFields();
    void project();

    float getDeltatime() const { return deltaT; }

    Eigen::SparseMatrix<float> generateLaplacian(Eigen::Vector3i gridSize);

    const Grid& getGrid() const { return grid; }
};