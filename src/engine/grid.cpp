#include "grid.hpp"

#include <cmath>
#include <iostream>
#include <algorithm>

namespace engine {

/*
    Initializes the grid with 0

    @param gridSize
        3D-Vector for width, height and depth of the discretization grid
    
*/
void Grid::initialize(Eigen::Vector3i _gridSize, std::array<float, 6> _boundaryCondition) {
    nx = _gridSize.x();
    ny = _gridSize.y();
    nz = _gridSize.z();

    cellType = std::vector<CellType>(nx * ny * nz, CellType::Fluid);

    u = std::vector<float>((nx+1) * ny * nz, 0);
    v = std::vector<float>(nx * (ny+1) * nz, 0);
    w = std::vector<float>(nx * ny * (nz+1), 0);

    uType = std::vector<FaceType>((nx+1) * ny * nz, FaceType::Fluid_Fluid);
    vType = std::vector<FaceType>(nx * (ny+1) * nz, FaceType::Fluid_Fluid);
    wType = std::vector<FaceType>(nx * ny * (nz+1), FaceType::Fluid_Fluid);
    
    for(auto& scalarField : scalarFields) {
        scalarField = {
            std::vector<float>(nx * ny * nz, 0),
            false
        };
    }

    for(int i = 0; i < _boundaryCondition.size(); i++) {
        Plane plane = getBoundaryPlane(static_cast<Direction>(i));
        if(_boundaryCondition[i] < 1e-8) {
            forEachFace([](FaceView face) {
                face.type = FaceType::Fluid_Solid;
            }, [&plane](FaceView face) {
                return face.isInPlane(plane);
            });
        } else {
            forEachFace([](FaceView face) {
                face.type = FaceType::Fluid_Fluid;
            }, [&plane](FaceView face) {
                return face.isInPlane(plane);
            });
        }
    }

    scalarFields[ScalarFieldID::Smoke].advect = true;

    dx = dy = dz = 0.3;
    // dy = 1.0/float(ny);
    // dz = 1.0/float(nz);
}

void Grid::moveCoord(Direction dir, GridCoord& coord, float magnitude) const {
    Eigen::Vector3f vec = getBasisVector(dir) * magnitude;
    coord.x += vec.x();
    coord.y += vec.y();
    coord.z += vec.z();
}

float lerp(float x, float y, float t) {
    return x + (y-x) * t;
}

float biLerp(float ul, float ur, float bl, float br, Eigen::Vector2f pos) {
    float uInt = lerp(ul, ur, pos.x());
    float bInt = lerp(bl, br, pos.x());
    return lerp(uInt, bInt, pos.y());
}

float triLerp(
    float ulf, float urf, float blf, float brf, // front square
    float ulb, float urb, float blb, float brb, // back square
    Eigen::Vector3f pos
) {
    float fInt = biLerp(ulf, urf, blf, brf, {pos.x(), pos.y()});
    float bInt = biLerp(ulb, urb, blb, brb, {pos.x(), pos.y()});
    return lerp(fInt, bInt, pos.z());
}

/*
    Returns the velocity of an arbitrary position in the volume.
*/
Eigen::Vector3f Grid::interpolateVelocity(GridCoord position) const {
    float interpolatedX = getVelocityU(position);
    float interpolatedY = getVelocityV(position);
    float interpolatedZ = getVelocityW(position);
    return {interpolatedX, interpolatedY, interpolatedZ};
}

float Grid::getExactU(Index3D idx) const {
    idx.i = std::clamp(idx.i, 0, nx);
    idx.j = std::clamp(idx.j, 0, ny-1);
    idx.k = std::clamp(idx.k, 0, nz-1);
    return u.at(uIndex(idx));
}

float Grid::getExactV(Index3D idx) const {
    idx.i = std::clamp(idx.i, 0, nx-1);
    idx.j = std::clamp(idx.j, 0, ny);
    idx.k = std::clamp(idx.k, 0, nz-1);
    return v.at(vIndex(idx));
}

float Grid::getExactW(Index3D idx) const {
    idx.i = std::clamp(idx.i, 0, nx-1);
    idx.j = std::clamp(idx.j, 0, ny-1);
    idx.k = std::clamp(idx.k, 0, nz);
    return w.at(wIndex(idx));
}

float Grid::getExactScalar(ScalarFieldID type, Index3D idx) const {
    idx.i = std::clamp(idx.i, 0, nx-1);
    idx.j = std::clamp(idx.j, 0, ny-1);
    idx.k = std::clamp(idx.k, 0, nz-1);
    return scalarFields[type].data.at(cellIndex(idx));
}

float Grid::getVelocityU(GridCoord coord) const {
    GridCoord shiftedCoord = {coord.x, coord.y - 0.5f, coord.z - 0.5f};
    int x = std::floor(shiftedCoord.x);
    int y = std::floor(shiftedCoord.y);
    int z = std::floor(shiftedCoord.z);

    Eigen::Vector3f localCoord = {shiftedCoord.x - float(x), shiftedCoord.y - float(y), shiftedCoord.z - float(z)};

    return triLerp(
        getExactU({x, y, z}), getExactU({x + 1, y, z}), getExactU({x, y + 1, z}), getExactU({x + 1, y + 1, z}),
        getExactU({x, y, z+1}), getExactU({x + 1, y, z+1}), getExactU({x, y + 1, z+1}), getExactU({x + 1, y + 1, z+1}),
        localCoord
    );
}

float Grid::getVelocityV(GridCoord coord) const {
    GridCoord shiftedCoord = {coord.x - 0.5f, coord.y, coord.z - 0.5f};
    int x = std::floor(shiftedCoord.x);
    int y = std::floor(shiftedCoord.y);
    int z = std::floor(shiftedCoord.z);

    Eigen::Vector3f localCoord = {shiftedCoord.x - float(x), shiftedCoord.y - float(y), shiftedCoord.z - float(z)};

    return triLerp(
        getExactV({x, y, z}), getExactV({x + 1, y, z}), getExactV({x, y + 1, z}), getExactV({x + 1, y + 1, z}),
        getExactV({x, y, z+1}), getExactV({x + 1, y, z+1}), getExactV({x, y + 1, z+1}), getExactV({x + 1, y + 1, z+1}),
        localCoord
    );
}

float Grid::getVelocityW(GridCoord coord) const {
    GridCoord shiftedCoord = {coord.x - 0.5f, coord.y - 0.5f, coord.z};
    int x = std::floor(shiftedCoord.x);
    int y = std::floor(shiftedCoord.y);
    int z = std::floor(shiftedCoord.z);

    Eigen::Vector3f localCoord = {shiftedCoord.x - float(x), shiftedCoord.y - float(y), shiftedCoord.z - float(z)};

    return triLerp(
        getExactW({x, y, z}), getExactW({x + 1, y, z}), getExactW({x, y + 1, z}), getExactW({x + 1, y + 1, z}),
        getExactW({x, y, z+1}), getExactW({x + 1, y, z+1}), getExactW({x, y + 1, z+1}), getExactW({x + 1, y + 1, z+1}),
        localCoord
    );
}

float Grid::getMaxVelocity() const {
    auto uit = std::max_element(u.begin(), u.end(), [](int a, int b) {
        return std::abs(a) < std::abs(b);
    });
    auto vit = std::max_element(v.begin(), v.end(), [](int a, int b) {
        return std::abs(a) < std::abs(b);
    });
    auto wit = std::max_element(w.begin(), w.end(), [](int a, int b) {
        return std::abs(a) < std::abs(b);
    });
    return std::max(std::max(std::abs(*uit), std::abs(*vit)), std::abs(*wit));
}

float Grid::getScalarField(ScalarFieldID type, GridCoord coord) const {
    GridCoord shiftedCoord = {coord.x - 0.5f, coord.y - 0.5f, coord.z - 0.5f};
    int x = std::floor(shiftedCoord.x);
    int y = std::floor(shiftedCoord.y);
    int z = std::floor(shiftedCoord.z);

    Eigen::Vector3f localCoord = {shiftedCoord.x - float(x), shiftedCoord.y - float(y), shiftedCoord.z - float(z)};

    return triLerp(
        getExactScalar(type, {x, y, z}), getExactScalar(type, {x + 1, y, z}), getExactScalar(type, {x, y + 1, z}), getExactScalar(type, {x + 1, y + 1, z}),
        getExactScalar(type, {x, y, z+1}), getExactScalar(type, {x + 1, y, z+1}), getExactScalar(type, {x, y + 1, z+1}), getExactScalar(type, {x + 1, y + 1, z+1}),
        localCoord
    );
}

float Grid::getDivergence(GridCoord coord) const {
    float dvdx = (getVelocityU({coord.x + 0.5f, coord.y, coord.z}) - getVelocityU({coord.x - 0.5f, coord.y, coord.z})) / dx;
    float dvdy = (getVelocityV({coord.x, coord.y + 0.5f, coord.z}) - getVelocityV({coord.x, coord.y - 0.5f, coord.z})) / dy;
    float dvdz = (getVelocityW({coord.x, coord.y, coord.z + 0.5f}) - getVelocityW({coord.x, coord.y, coord.z - 0.5f})) / dz;
    return dvdx + dvdy + dvdz;
}

float Grid::getScalarGradient(ScalarFieldID type, GridCoord coord, Axis axis) const {
    switch(axis) {
        case Axis::X: return getScalarGradX(type, coord);
        case Axis::Y: return getScalarGradY(type, coord);
        case Axis::Z: return getScalarGradZ(type, coord);
    } return getScalarGradX(type, coord);
}

float Grid::getScalarGradX(ScalarFieldID type, GridCoord coord) const {
    return (getScalarField(type, {coord.x + 0.5f, coord.y, coord.z}) - getScalarField(type, {coord.x - 0.5f, coord.y, coord.z})) / dx;
}

float Grid::getScalarGradY(ScalarFieldID type, GridCoord coord) const {
    return (getScalarField(type, {coord.x, coord.y + 0.5f, coord.z}) - getScalarField(type, {coord.x, coord.y - 0.5f, coord.z})) / dy;
}

float Grid::getScalarGradZ(ScalarFieldID type, GridCoord coord) const {
    return (getScalarField(type, {coord.x, coord.y, coord.z + 0.5f}) - getScalarField(type, {coord.x, coord.y, coord.z - 0.5f})) / dz;
}

}