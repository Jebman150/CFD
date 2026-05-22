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
void Grid::initialize(std::array<int, Axis::Dim> _n, std::array<float, 6> _boundaryCondition) {
    n = _n;
    for(int i = 0; i < Axis::Dim; i++) {
        totalCellCount *= n.at(i);
        d.at(i) = 0.3;
    }

    cellType = std::vector<CellType>(totalCellCount, CellType::Fluid);

    for(int i = 0; i < Axis::Dim; i++) {
        velocities[i] = std::vector<float>(getMACCellCount(static_cast<Axis>(i)), 0);
        faceTypes[i] = std::vector<FaceType>(getMACCellCount(static_cast<Axis>(i)), FaceType::Fluid_Fluid);
    }
    
    for(auto& scalarField : scalarFields) {
        scalarField = {
            std::vector<float>(totalCellCount, 0),
            false
        };
    }
    scalarFields[ScalarFieldID::Smoke].advect = true;
    // dy = 1.0/float(ny);
    // dz = 1.0/float(nz);
}

float lerp(float x, float y, float t) {
    return x + (y-x) * t;
}

float nLerp(
    const std::vector<float>& values,
    const std::vector<float>& pos,
    int dim
) {
    if (dim == 1) {
        return lerp(values[0], values[1], pos[0]);
    }

    size_t half = values.size() / 2;

    std::vector<float> lower(values.begin(), values.begin() + half);
    std::vector<float> upper(values.begin() + half, values.end());

    std::vector<float> subPos(pos.begin(), pos.end() - 1);

    float a = nLerp(lower, subPos, dim - 1);
    float b = nLerp(upper, subPos, dim - 1);

    return lerp(a, b, pos[dim - 1]);
}

/*
    Returns the velocity of an arbitrary position in the volume.
*/
Eigen::VectorXf Grid::interpolateVelocity(GridCoord position) const {
    Eigen::VectorXf result(Axis::Dim);
    for(int i = 0; i < Axis::Dim; i++) {
        result(i) = getVelocity(static_cast<Axis>(i), position);
    }
    return result;
}

float Grid::getExactVelocity(Axis axis, MultiIndex idx) const {
    if(!idx.isValid()) return 0;
    return velocities[axis].at(idx.get());
}

float Grid::getExactScalar(ScalarFieldID type, MultiIndex idx) const {
    if(!idx.isValid()) return 0;
    return scalarFields[type].data.at(idx.get());
}

float Grid::getVelocity(Axis axis, GridCoord coord) const {
    GridCoord shiftedCoord = coord.shift(axis);
    MultiIndex idx(getMACSize(axis));
    std::array<int, Axis::Dim> flooredPos;
    std::vector<float> localCoord;
    for(int i = 0; i < Axis::Dim; i++) {
        flooredPos.at(i) = std::floorf(coord.coord[i]);
        idx.advance(static_cast<Axis>(i), flooredPos.at(i));
        localCoord.emplace_back(shiftedCoord.coord[i] - float(localCoord[i]));
    }

    std::vector<float> samplePoints;
    samplePoints.reserve(1 << Axis::Dim);
    for (int mask = 0; mask < (1 << Axis::Dim); mask++) {
        MultiIndex target = idx;
        for (int axis = 0; axis < Axis::Dim; axis++) {
            if (mask & (1 << axis)) {
                target.advance(static_cast<Axis>(axis), 1);
            }
        }
        samplePoints.emplace_back(
            getExactVelocity(axis, target)
        );
    }

    return nLerp(
        samplePoints,
        localCoord,
        Axis::Dim
    );
}

float Grid::getMaxVelocity() const {
    float max = 0;
    for(int i = 0; i < Axis::Dim; i++) {
        auto uit = std::max_element(velocities[i].begin(), velocities[i].end(), [](int a, int b) {
            return std::abs(a) < std::abs(b);
        });
        if(max < std::abs(*uit)) max = std::abs(*uit);
    }
    return max;
}

float Grid::getScalarField(ScalarFieldID type, GridCoord coord) const {
    GridCoord shiftedCoord = coord.shift();
    MultiIndex idx(getSize());
    std::array<int, Axis::Dim> flooredPos;
    std::vector<float> localCoord;
    for(int i = 0; i < Axis::Dim; i++) {
        flooredPos.at(i) = std::floorf(coord.coord[i]);
        idx.advance(static_cast<Axis>(i), flooredPos.at(i));
        localCoord.emplace_back(shiftedCoord.coord[i] - float(localCoord[i]));
    }

    std::vector<float> samplePoints;
    samplePoints.reserve(1 << Axis::Dim);
    for (int mask = 0; mask < (1 << Axis::Dim); mask++) {
        MultiIndex target = idx;
        for (int axis = 0; axis < Axis::Dim; axis++) {
            if (mask & (1 << axis)) {
                target.advance(static_cast<Axis>(axis), 1);
            }
        }
        samplePoints.emplace_back(
            getExactScalar(type, target)
        );
    }

    return nLerp(
        samplePoints,
        localCoord,
        Axis::Dim
    );
}

float Grid::getDivergence(MultiIndex idx) const {
    float div = 0;
    for(int i = 0; i < Axis::Dim; i++) {
        div += (getExactVelocity(static_cast<Axis>(i), idx.getSucceeding(static_cast<Axis>(i))) - getExactVelocity(static_cast<Axis>(i), idx)) / d.at(i);
    }
    return div;
}

float Grid::getScalarGradient(ScalarFieldID type, MultiIndex idx, Axis axis) const {
    return (getExactScalar(type, idx.getSucceeding(axis)) - getExactScalar(type, idx)) / d.at(axis);
}

/*float Grid::getScalarGradient(ScalarFieldID type, GridCoord coord, Axis axis) const {
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
}*/

}