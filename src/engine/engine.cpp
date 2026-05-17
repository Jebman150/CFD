#include "engine.hpp"

#include <Eigen/Sparse>
#include <iostream>

/*
    Initializes the grid
*/
void Engine::initSim() {
    deltaT = 0.001;
    grid.initialize(Eigen::Vector2i(15, 15)); // Non-square grids currently bug
}

/*
    TODO: adjust time step in each simulation step
*/
void Engine::adjustTimestep() {
    return;
}

void Engine::applyBoundaryCondition() {
    //Set boundary x velocities to partial in-/outflow
    for(int i = 0; i < grid.getHeight(); i++) {
        Index2D idx = {0, i};
        if(i == 0 || i == (grid.getHeight() - 1)) grid.setVelocityU(idx, 0);
        else grid.setVelocityU(idx, 0);

        idx.i = grid.getWidth();
        if(i == 0 || i == (grid.getHeight() - 1)) grid.setVelocityU(idx, 0);
        else grid.setVelocityU(idx, 0);
    }

    //Set boundary y velocities to 0
    for(int i = 0; i < grid.getWidth(); i++) {
        Index2D idx = {i, 0};
        grid.setVelocityV(idx, 0);
        idx.j = grid.getHeight();
        grid.setVelocityV(idx, 0);
    }
}

/*
    Moves velocities over the grid.

    1. Get previous position
    2. Update selected velocity to value at prev. pos.
*/
void Engine::advect() {
    grid.flushBuffer();

    //Update x velocities
    for(int i = 1; i < grid.getWidth(); i++) {
        for(int j = 0; j < grid.getHeight(); j++) {
            GridCoord coord = grid.coordFromUIndex({i, j});
            Eigen::Vector2d currentVelocity = grid.interpolateVelocity(coord);
            auto oldPosVec = Eigen::Vector2d{coord.x, coord.y} - currentVelocity * deltaT / grid.getDx();
            GridCoord oldCoord = {oldPosVec.x(), oldPosVec.y()};
            grid.bufferVelocityU({i, j}, grid.interpolateVelocity(oldCoord).x());
        }
    }

    //Update y velocities
    for(int i = 0; i < grid.getWidth(); i++) {
        for(int j = 1; j < grid.getHeight(); j++) {
            GridCoord coord = grid.coordFromVIndex({i, j});
            Eigen::Vector2d currentVelocity = grid.interpolateVelocity(coord);
            auto oldPosVec = Eigen::Vector2d{coord.x, coord.y} - currentVelocity * deltaT / grid.getDy();
            GridCoord oldCoord = {oldPosVec.x(), oldPosVec.y()};
            grid.bufferVelocityV({i, j}, grid.interpolateVelocity(oldCoord).y());
        }
    }
    
    grid.applyBuffer();
}

/*
    Computes the pressure of the current state of the grid per Modified Incomplete Cholesky Conjugate Gradient, Level Zero.
    Updates the vertical and horizontal velocities with the computed pressure gradient afterwards.
*/
void Engine::project() {
    //Add artificial movement in grid
    Index2D artificialMovCoord = {std::round(grid.getWidth() / 2.0), std::round(grid.getHeight() / 2.0)};
    grid.setVelocityV(artificialMovCoord, 3 * sin(60.0 * currentT));

    int width = grid.getWidth();
    int height = grid.getHeight();
    int n = width * height;

    double cx = 1/(grid.getDx() * grid.getDx());
    double cy = 1/(grid.getDy() * grid.getDy());

    Eigen::SparseMatrix<double> A(n, n);
    for(int i = 0; i < n; i++) {
        int numOfcx = 0;
        int numOfcy = 0;
        if((i + height) < n) {
            A.insert(i, i + height) = -cx;
            numOfcx++;
        }
        if((i - height) >= 0) {
            A.insert(i, i - height) = -cx;
            numOfcx++;
        }
        if((i + 1) % height != 0) {
            A.insert(i, i + 1) = -cy;
            numOfcy++;
        }
        if(i % height != 0 && i != 0) {
            A.insert(i, i - 1) = -cy;
            numOfcy++;
        }
        A.insert(i, i) = numOfcx * cx + numOfcy * cy;
    }

    //std::cout << Eigen::MatrixXd(A) << std::endl;

    Eigen::VectorXd d(n);
    int counter = 0;
    for(double x = 0.5; x < width; x++) {
        for(double y = 0.5; y < height; y++) {
            /*std::cout << "Get divergence (" << x << ", " << y << ")" << std::endl
                << " | (" << x+0.5 << ", " << y << "): " << grid.getVelocityU({x + 0.5, y}) << std::endl
                << " | (" << x-0.5 << ", " << y << "): " << grid.getVelocityU({x - 0.5, y}) << std::endl
                << " | (" << x << ", " << y + 0.5 << "): " << grid.getVelocityV({x, y + 0.5}) << std::endl
                << " | (" << x << ", " << y - 0.5 << "): " << grid.getVelocityV({x, y - 0.5}) << std::endl
                << std::endl;*/
            d(counter++) = grid.getDivergence({x, y});
        }
    }
    d = d * (density / deltaT);
    d.array() -= d.mean();
    //std::cout << "Sum: " << d.sum() << std::endl;
    //std::cout << d << std::endl;

    auto p = pressureSolver.computePressure(d, A);

    //std::cout << "Check: " << A*p << std::endl;

    counter = 0;
    for(int i = 0; i < width; i++) {
        for(int j = 0; j < height; j++) {
            grid.setPressure({i, j}, p(counter++));
        }
    }

    //Update velocities in x direction
    for(int i = 1; i < width; i++) {
        for(int j = 0; j < height; j++) {
            GridCoord coord = grid.coordFromUIndex({i, j});
            grid.updateU({i, j}, (deltaT / density) * ((grid.getPressure({coord.x + 0.5, coord.y}) - grid.getPressure({coord.x - 0.5, coord.y})) / grid.getDx()));
        }
    }

    //Update velocities in y direction
    for(int i = 0; i < width; i++) {
        for(int j = 1; j < height; j++) {
            GridCoord coord = grid.coordFromVIndex({i, j});
            grid.updateV({i, j}, (deltaT / density) * ((grid.getPressure({coord.x, coord.y + 0.5}) - grid.getPressure({coord.x, coord.y - 0.5})) / grid.getDy()));
        }
    }

    currentT += deltaT;
}