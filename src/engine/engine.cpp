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
    boundaryConditions = {1.f, 1.f, 0.f, 0.f, 0.f, 0.f};
    
    float sum = 0;
    bool inflow = true;
    for(auto cond : boundaryConditions) {
        sum += inflow ? cond : -cond;
        inflow = !inflow;
    }
    if(std::abs(sum) > 1e-8) std::cout << "WARNING: Invalid boundary conditions!" << std::endl;

    grid.initialize(
        gridSize,
        boundaryConditions
    );
}

/*
    TODO: adjust time step in each simulation step
*/
void Engine::adjustTimestep() {
    float maxU = std::max(grid.getMaxVelocity(), 0.1f);
    deltaT = cTarget * grid.getDx(Axis::X) / maxU;
}

void Engine::checkCFDCondition() {
    float maxU = grid.getMaxVelocity();
    float C = maxU * deltaT / grid.getDx(Axis::X);
    if(C < 0.1 || C > 1) {
        std::cout << "WARNING: CFD Condition " << C << std::endl;
    }
    std::cout << "CFD: " << C << std::endl;
}

void Engine::applyBoundaryCondition() {
    // Internal faces
    grid.forEachFace([](FaceView face){
        face.value = 0;
    }, [](FaceView face) {
        return face.type == FaceType::Solid_Solid || face.type == FaceType::Fluid_Solid || face.type == FaceType::Solid_Fluid;
    });

    // Domain boundary flow
    for(int i = 0; i < Axis::Dim; i++) {
        float value = boundaryConditions[i*2];
        grid.forEachFace(grid.getBoundaryPlane(static_cast<Axis>(i), false), [value](FaceView face){
            face.value = value;
            if(std::abs(value) > 1e-8) face.type = Fluid_Fluid;
        });
        float value = boundaryConditions[i*2+1];
        grid.forEachFace(grid.getBoundaryPlane(static_cast<Axis>(i), true), [value](FaceView face){
            face.value = value;
            if(std::abs(value) > 1e-8) face.type = Fluid_Fluid;
            else face.type = Fluid_Solid;
        });
    }
}

void Engine::setSolidCells() {
    GridCoord center;
    for(int i = 0; i < Axis::Dim; i++) {
        center.coord[i] = grid.getSize(static_cast<Axis>(i));
    }
    float radius = 4;

    grid.forEachCell([this](CellView cell) {
        cell.type = CellType::Solid;
    }, [&radius, &center](CellView cell) {
        return center.dist(cell.coord) < radius;
    });
}

void Engine::updateFaceTypes() {
    grid.updateFaceTypes();
}

void Engine::spawnSmoke() {
    GridCoord center;
    for(int i = 0; i < Axis::Dim; i++) {
        center.coord[i] = grid.getSize(static_cast<Axis>(i));
    }
    int height = 4;

    for(MultiIndex idx(grid.getSize()); !idx.overflow(); idx++) {
        GridCoord coord(idx);
        if(!coord.coord[0] == 1) continue;
        if(!(std::abs(coord.coord[1]-center.coord[1]) < height)) continue;

        grid.setScalarField(ScalarFieldID::Smoke, idx, 1.f);
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
    std::array<std::vector<float>, Axis::Dim> velocityBuffer;
    for(int i = 0; i < Axis::Dim; i++) {
        int size = 0;
        auto dims = grid.getMACSize(static_cast<Axis>(i));
        for(int i = 0 ; i < Axis::Dim; i++) {
            size += dims[i];
        }
        velocityBuffer.at(i) = std::vector<float>(size, 0);
    }

    grid.forEachCell([&velocityBuffer, this] (FaceView& face) {
        Eigen::VectorXf currentVelocity = grid.interpolateVelocity(face.coord);
        Eigen::VectorXf pos = face.coord;
        GridCoord oldCoord = GridCoord(pos - currentVelocity * deltaT);
        velocityBuffer[face.axis].at(face.idx.get()) = grid.interpolateVelocity(oldCoord)[face.axis];
    }, [this] (FaceView& face) {
        return !face.isInPlane(grid.getBoundaryPlane(face.axis, false)) && !face.isInPlane(grid.getBoundaryPlane(face.axis, true));
    });

    grid.overrideVelocities(velocityBuffer);
}

void Engine::advectScalarFields() {
    for(int id = 0; id < static_cast<int>(ScalarFieldID::NumFields); id++) {
        auto& field = grid.getScalarField(static_cast<ScalarFieldID>(id));
        if(!field.advect) continue;

        std::vector<float> buffer(grid.getC);

        for(MultiIndex idx(grid.getSize()); !idx.overflow(); idx++) {
            GridCoord coord(idx);
            Eigen::VectorXf currentVelocity = grid.interpolateVelocity(coord);
            Eigen::VectorXf pos = coord;
            GridCoord oldCoord = GridCoord(pos - currentVelocity * deltaT);
            buffer.at(idx.get()) = grid.getScalarField(static_cast<ScalarFieldID>(id), oldCoord);
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
    //MultiIndex artificialMovCoord = {std::round(grid.getWidth() / 2.0), std::round(grid.getHeight() / 2.0), 0};
    //grid.setVelocityV(artificialMovCoord, 300 * sin(3 * currentT));
    //grid.setVelocityU(artificialMovCoord, 300 * cos(3 * currentT));
    //grid.forEachFace([](FaceView face) {
    //    face.value = 1.f;
    //}, [](FaceView face) {
    //    return face.axis == Axis::X && face.idx.i == 1 && std::fabs(face.idx.j - 7.f) < 4;
    //});
    //grid.setVelocityU({1, 7, 0}, 1.f);

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
    if(std::abs(d.sum()) > 0.1) {
        std::cout << "WARNING: Problem singular!" << std::endl;
        std::cout << "Sum of divergence: " << std::abs(d.sum()) << std::endl;
    }

    Eigen::VectorXf p = pressureSolver.computePressure(d);

    counter = 0;
    for(Indexer3D idxer(grid.getWidth(), grid.getHeight(), grid.getDepth()); !idxer.end(); idxer++) {
        grid.setScalarField(ScalarFieldID::Pressure, idxer.get(), p(counter++));
    }

    for(int ax = 0; ax < Axis::Dim; ax++) {
        Axis axis = static_cast<Axis>(ax);
        for(int i = 1; i < grid.getSize(axis); i++) {
            Plane plane = {axis, i};
            grid.forEachFace(plane, [this](FaceView face) {
                GridCoord coord = face.coord;
                face.value -= (deltaT / density) * (grid.getScalarGradient(ScalarFieldID::Pressure, coord, face.axis));
            });
        }
    }

    currentT += deltaT;
}

}