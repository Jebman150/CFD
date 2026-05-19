#include "engine.hpp"

#include <Eigen/Sparse>
#include <iostream>

namespace engine {

/*
    Initializes the grid
*/
void Engine::initSim() {
    deltaT = 0.05;
    Eigen::Vector3i gridSize = {50, 50, 1};

    grid.initialize(
        gridSize,
        {Wall, Wall, Wall, Wall, Wall, Wall}
    );

    pressureSolver.initialize(gridSize, {grid.getDx(), grid.getDy(), grid.getDz()});
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
        vel *= boundaryVelocity;
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

void Engine::spawnSmoke() {
    Eigen::Vector3f center = {grid.getWidth()/2.f, grid.getHeight()/2.f, grid.getDepth()/2.f};
    float radius = 2;
    for(int k = 0; k < grid.getDepth(); k++) {
        for(int j = 0; j < grid.getHeight(); j++) {
            for(int i = 0; i < grid.getWidth(); i++) {
                if(std::abs(i - center.x()) < 4 && std::abs(j - center.y()) < 4 && std::abs(k - center.z()) < 4) {
                    grid.setScalarField(ScalarFieldID::Smoke, {i, j, k}, 1.f);
                }
            }
        }
    }
}

/*
    Moves velocities over the grid.

    1. Get previous position
    2. Update selected velocity to value at prev. pos.
*/
void Engine::advect() {
    advectVelocities();
    advectScalarFields();
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
    for(int id = 0; id < static_cast<int>(ScalarFieldID::NumFields); id++) {
        auto& field = grid.getScalarField(static_cast<ScalarFieldID>(id));
        if(!field.advect) continue;

        std::vector<float> buffer(grid.getWidth() * grid.getHeight() * grid.getDepth());

        for(int k = 0; k < grid.getDepth(); k++) {
            for(int j = 0; j < grid.getHeight(); j++) {
                for(int i = 0; i < grid.getWidth(); i++) {
                    GridCoord coord = grid.coordFromCellIndex({i, j, k});
                    Eigen::Vector3d currentVelocity = grid.interpolateVelocity(coord);
                    Eigen::Vector3d oldPosVec = Eigen::Vector3d{coord.x, coord.y, coord.z} - currentVelocity * deltaT;
                    GridCoord oldCoord = {oldPosVec.x(), oldPosVec.y(), oldPosVec.z()};
                    buffer.at(grid.cellIndex({i, j, k})) = grid.getScalarField(static_cast<ScalarFieldID>(id), oldCoord);
                }
            }
        }

        grid.overrideScalarField(static_cast<ScalarFieldID>(id), buffer);
    }
}

/*
    Computes the pressure of the current state of the grid per Modified Incomplete Cholesky Conjugate Gradient, Level Zero.
    Updates the vertical and horizontal velocities with the computed pressure gradient afterwards.
*/
void Engine::project() {
    //Add artificial movement in grid
    Index3D artificialMovCoord = {std::round(grid.getWidth() / 2.0), std::round(grid.getHeight() / 2.0), 0};
    grid.setVelocityV(artificialMovCoord, 300 * sin(3 * currentT));
    grid.setVelocityU(artificialMovCoord, 300 * cos(3 * currentT));

    int width = grid.getWidth();
    int height = grid.getHeight();
    int depth = grid.getDepth();
    int n = width * height * depth;

    Eigen::VectorXf d(n);
    int counter = 0;
    for(float z = 0.5f; z < depth; z++) {
        for(float y = 0.5f; y < height; y++) {
            for(float x = 0.5f; x < width; x++) {
                d(counter++) = grid.getDivergence({x, y, z});
            }
        }
    }
    d = d * (density / deltaT);
    d.array() -= d.mean();
    //if(std::abs(d.sum()) > 0.1) {
    //    std::cout << "WARNING: Problem singular!" << std::endl;
    //    std::cout << "Sum of divergence: " << std::abs(d.sum()) << std::endl;
    //}

    Eigen::VectorXf p = pressureSolver.computePressure(d);

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
                face.value -= (deltaT / density) * (grid.getScalarGradient(ScalarFieldID::Pressure, coord, face.axis));
            });
        }
    }

    currentT += deltaT;
}

}