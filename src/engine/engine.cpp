#include "engine.hpp"

/*
    Initializes the grid
*/
void Engine::initSim() {
    grid.initialize(Eigen::Vector2d(10, 10));
}