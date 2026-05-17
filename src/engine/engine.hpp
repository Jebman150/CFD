#pragma once

#include "grid.hpp"
#include "pressureSolver.hpp"

class Engine {
    Grid grid;
    PressureSolver pressureSolver;

    double deltaT;

    double density = 1000.f;
    double currentT = 0;
public:
    void initSim();
    void adjustTimestep();
    void applyBoundaryCondition();
    void advect();
    void project();

    const Grid& getGrid() const { return grid; }
};