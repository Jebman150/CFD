#pragma once

#include "grid.hpp"
#include "solver.hpp"
#include "linearOperator.hpp"

namespace engine {

class Engine {
    Grid grid;
    std::unique_ptr<Solver> solver;
    LaplacianOperatorGPU laplacian;

    float deltaT;

    float cTarget = 2;
    float density = 1000.f;
    float currentT = 0;
    std::array<float, 6> boundaryConditions;
public:
    void initSim();
    void adjustTimestep();
    void checkCFDCondition();
    void setSolidCells();
    void updateFaceTypes();
    void spawnSmoke();
    void applyBoundaryCondition();
    void advect();
    void advectVelocities();
    void advectScalarFields();
    void project();

    float getDeltatime() const { return deltaT; }

    const Grid& getGrid() const { return grid; }
};

}