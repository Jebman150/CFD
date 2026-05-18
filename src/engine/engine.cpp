#include "engine.hpp"

#include <Eigen/Sparse>
#include <iostream>

/*
    Initializes the grid
*/
void Engine::initSim() {
    deltaT = 0.05;
    Eigen::Vector3i gridSize = {20, 20, 1};

    grid.initialize(
        gridSize,
        {Wall, partialOutflow, partialInflow, Wall, Wall, Wall}
    );

    laplacian = generateLaplacian(gridSize);
    pressureSolver.compute(laplacian);
}

/*
    TODO: adjust time step in each simulation step
*/
void Engine::adjustTimestep() {
    float maxU = std::max(grid.getMaxVelocity(), 0.1f);
    deltaT = cTarget * grid.getDx() / maxU;
}

void Engine::checkCFDCondition() {
    float maxU = grid.getMaxVelocity();
    float C = maxU * deltaT / grid.getDx();
    if(C < 0.1 || C > 1) {
        std::cout << "WARNING: CFD Condition " << C << std::endl;
    }
    std::cout << "CFD: " << C << std::endl;
}

void Engine::applyBoundaryCondition() {
    for(int dir = 0; dir < Direction::NUM; dir++) {
        Direction direction = static_cast<Direction>(dir);
        auto condition = grid.getBoundaryCondition(direction);
        if(condition == Open) continue;

        float inflowSign = (dir%2 == 0) ? 1 : -1;
        float vel = (condition == Wall) ? 0 : ((condition == Inflow || condition == partialInflow) ? inflowSign : -inflowSign);
        auto plane = grid.getBoundaryPlane(direction);
        grid.forEachVelocity(plane, [this, vel, condition](FaceView face){
            if(condition != partialInflow && condition != partialOutflow) {
                face.value = vel;
            } else {
                switch(face.axis) {
                    case Axis::X: face.value = (face.idx.j == 0 || face.idx.j == grid.getHeight()-1) ? 0 : vel; break;
                    case Axis::Y: face.value = (face.idx.i == 0 || face.idx.i == grid.getWidth()-1) ? 0 : vel; break;
                    case Axis::Z: face.value = (face.idx.j == 0 || face.idx.j == grid.getHeight()-1) ? 0 : vel; break;
                }
                
            }
        });
    }
}

/*
    Moves velocities over the grid.

    1. Get previous position
    2. Update selected velocity to value at prev. pos.
*/
void Engine::advect() {
    advectVelocities();
}

void Engine::advectVelocities() {
    std::array<std::vector<float>, Axis::Dim> velocityBuffer = {
        std::vector<float>((grid.getWidth()+1) * grid.getHeight() * grid.getDepth()),
        std::vector<float>(grid.getWidth() * (grid.getHeight()+1) * grid.getDepth()),
        std::vector<float>(grid.getWidth() * grid.getHeight() * (grid.getDepth()+1))
    };

    for(int ax = 0; ax < Axis::Dim; ax++) {
        Axis axis = static_cast<Axis>(ax);
        for(int i = 1; i < grid.getSize(axis); i++) {
            Plane plane = {axis, i};
            grid.forEachVelocity(plane, [this, &velocityBuffer, ax](FaceView face) {
                Eigen::Vector3d currentVelocity = grid.interpolateVelocity(face.coord);
                Eigen::Vector3d oldPosVec = Eigen::Vector3d{face.coord.x, face.coord.y, face.coord.z} - currentVelocity * deltaT;
                GridCoord oldCoord = {oldPosVec.x(), oldPosVec.y(), oldPosVec.z()};
                velocityBuffer[ax].at(grid.getVelocityIndex(face.axis, face.idx)) = grid.interpolateVelocity(oldCoord)[ax];
            });
        }
    }

    grid.overrideVelocities(velocityBuffer);
}

void Engine::advectScalarFields() {

}

/*
    Computes the pressure of the current state of the grid per Modified Incomplete Cholesky Conjugate Gradient, Level Zero.
    Updates the vertical and horizontal velocities with the computed pressure gradient afterwards.
*/
void Engine::project() {
    //Add artificial movement in grid
    Index3D artificialMovCoord = {std::round(grid.getWidth() / 2.0), std::round(grid.getHeight() / 2.0), 0};
    grid.setVelocityV(artificialMovCoord, 3 * sin(currentT));

    int width = grid.getWidth();
    int height = grid.getHeight();
    int depth = grid.getDepth();
    int n = width * height * depth;

    Eigen::VectorXf d(n);
    int counter = 0;
    for(float z = 0.5f; z < depth; z++) {
        for(float y = 0.5f; y < height; y++) {
            for(float x = 0.5f; x < width; x++) {
                /*std::cout << "Get divergence (" << x << ", " << y << ")" << std::endl
                    << " | (" << x+0.5f << ", " << y << "): " << grid.getVelocityU({x + 0.5f, y}) << std::endl
                    << " | (" << x-0.5f << ", " << y << "): " << grid.getVelocityU({x - 0.5f, y}) << std::endl
                    << " | (" << x << ", " << y + 0.5f << "): " << grid.getVelocityV({x, y + 0.5f}) << std::endl
                    << " | (" << x << ", " << y - 0.5f << "): " << grid.getVelocityV({x, y - 0.5f}) << std::endl
                    << std::endl;*/
                d(counter++) = grid.getDivergence({x, y, z});
            }
        }
    }
    d = d * (density / deltaT);
    d.array() -= d.mean();

    Eigen::VectorXf p = pressureSolver.solve(d);

    counter = 0;
    for(int k = 0; k < depth; k++) {
        for(int j = 0; j < height; j++) {
            for(int i = 0; i < width; i++) {
                grid.setScalarField(ScalarFieldID::Pressure, {i, j, k}, p(counter++));
            }
        }
    }

    for(int ax = 0; ax < Axis::Dim; ax++) {
        Axis axis = static_cast<Axis>(ax);
        for(int i = 1; i < grid.getSize(axis); i++) {
            Plane plane = {axis, i};
            grid.forEachVelocity(plane, [this](FaceView face) {
                GridCoord coord = face.coord;
                face.value += (deltaT / density) * (grid.getScalarGradient(ScalarFieldID::Pressure, coord, face.axis));
            });
        }
    }

    currentT += deltaT;
}

Eigen::SparseMatrix<float> Engine::generateLaplacian(Eigen::Vector3i gridSize) {
    int width  = gridSize.x();
    int height = gridSize.y();
    int depth  = gridSize.z();

    int n = width * height * depth;

    float cx = width * width;
    float cy = height * height;
    float cz = depth * depth;

    std::vector<Eigen::Triplet<float>> T;
    for (int i = 0; i < n; ++i)
    {
        int x = i % width;
        int y = (i / width) % height;
        int z = i / (width * height);

        float diag = 0.0;

        //X
        if (x > 0) {
            T.emplace_back(i, i - 1, -cx);
            diag += cx;
        }
        if (x < width - 1) {
            T.emplace_back(i, i + 1, -cx);
            diag += cx;
        }

        // Y
        if (y > 0) {
            T.emplace_back(i, i - width, -cy);
            diag += cy;
        }
        if (y < height - 1) {
            T.emplace_back(i, i + width, -cy);
            diag += cy;
        }

        // Z
        if (z > 0) {
            T.emplace_back(i, i - width * height, -cz);
            diag += cz;
        }
        if (z < depth - 1) {
            T.emplace_back(i, i + width * height, -cz);
            diag += cz;
        }

        T.emplace_back(i, i, diag);
    }

    Eigen::SparseMatrix<float> A(n,n);
    A.setFromTriplets(T.begin(), T.end());
    return A;
}