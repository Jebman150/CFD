#pragma once

#include "grid.hpp"
#include "pressureSolver.hpp"

class Engine {
    Grid grid;
    PressureSolver pressureSolver;

    float deltaT;

    float cTarget = 2;
    float density = 1000.f;
    float currentT = 0;
    float boundaryVelocity = 5.f;
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

    const Grid& getGrid() const { return grid; }
};