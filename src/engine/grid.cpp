#include "grid.hpp"

/*
    Initializes the grid with random values

    @param gridSize
        2D-Vector for width and height of the discretization grid
    
*/
void Grid::initialize(Eigen::Vector2d _gridSize) {
    gridSize = _gridSize;
    velocityX = Eigen::MatrixXd::Random(gridSize.x() + 1, gridSize.y());
    velocityY = Eigen::MatrixXd::Random(gridSize.x(), gridSize.y() + 1);
    pressure = Eigen::MatrixXd::Random(gridSize.x(), gridSize.y());
}