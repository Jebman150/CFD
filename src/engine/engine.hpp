#pragma once

#include "grid.hpp"

class Engine {
    Grid grid;

public:
    void initSim();

    const Grid& getGrid() const { return grid; }
};