#include "grid.hpp"

#include <cmath>
#include <algorithm>
#include <vector>
#include <cmath>

namespace engine {

/*
    Initializes the grid with 0

    @param gridSize
        3D-Vector for width, height and depth of the discretization grid
    
*/
void Grid::initialize(std::vector<int> _n, std::array<float, 6> _boundaryCondition) {
    n = _n;
    d = std::vector<float>(n.size());
    totalCellCount = 1;
    for(int i = 0; i < n.size(); i++) {
        totalCellCount *= n.at(i);
        d.at(i) = 0.3;
    }

    cellType = Field<CellType>(getSize(), CellType::Fluid);

    for(int i = 0; i < n.size(); i++) {
        velocities[i] = Field<float>(getMACSize(static_cast<Axis>(i)), 0);
        faceTypes[i] = Field<FaceType>(getMACSize(static_cast<Axis>(i)), FaceType::Fluid_Fluid);
    }
    
    for(auto& scalarField : scalarFields) {
        scalarField = Field<float>(getSize(), 0);
    }
    scalarFields[ScalarFieldID::Smoke].advect = true;
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
    Eigen::VectorXf result(n.size());
    for(int i = 0; i < n.size(); i++) {
        result(i) = getVelocity(static_cast<Axis>(i), position);
    }
    return result;
}

float Grid::getExactVelocity(Axis axis, MultiIndex idx) const {
    if(!idx.isValid()) return 0;
    return velocities[axis].at(idx);
}

float Grid::getExactScalar(ScalarFieldID type, MultiIndex idx) const {
    if(!idx.isValid()) return 0;
    return scalarFields[type].at(idx);
}

float Grid::getVelocity(Axis axis, GridCoord coord) const {
    //std::cout << "Sampling velocity at " << coord.coord[0] << ", " << coord.coord[1] << std::endl;
    //std::cout << "In direction " << axis << std::endl;
    GridCoord shiftedCoord = coord.shift(axis);
    //std::cout << " | Shifted to " << shiftedCoord.coord[0] << ", " << shiftedCoord.coord[1] << std::endl;
    std::vector<int> flooredPos = std::vector<int>(n.size());
    std::vector<float> localCoord = std::vector<float>(n.size());
    for(int i = 0; i < n.size(); i++) {
        flooredPos.at(i) = std::floorf(shiftedCoord.coord[i]);
        localCoord.at(i) = shiftedCoord.coord[i] - float(flooredPos[i]);
    }
    //std::cout << " | Floored to " << flooredPos[0] << ", " << flooredPos[1] << std::endl;
    //std::cout << " | Local coord " << localCoord[0] << ", " << localCoord[1] << std::endl;

    std::vector<float> samplePoints;
    samplePoints.reserve(1 << n.size());
    for (int mask = 0; mask < (1 << n.size()); mask++) {
        std::vector<int> indices = flooredPos;
        for (int i = 0; i < n.size(); i++) {
            if (mask & (1 << i)) {
                indices[i]++;
            }
            indices[i] = std::clamp(indices[i], 0, getMACSize(axis)[i]-1);
        }

        //std::cout << " | Adding sample at " << indices[0] << ", " << indices[1] << std::endl;
        MultiIndex target(indices, velocities[axis].getContext());
        //std::cout << " | Value: " << getExactVelocity(axis, target) << std::endl;
        samplePoints.emplace_back(
            getExactVelocity(axis, target)
        );
    }

    return nLerp(
        samplePoints,
        localCoord,
        n.size()
    );
}

float Grid::getMaxVelocity() const {
    float max = 0;
    for(int i = 0; i < n.size(); i++) {
        auto uit = std::max_element(velocities[i].getData().begin(), velocities[i].getData().end(), [](int a, int b) {
            return std::abs(a) < std::abs(b);
        });
        if(max < std::abs(*uit)) max = std::abs(*uit);
    }
    return max;
}

float Grid::getScalarField(ScalarFieldID type, GridCoord coord) const {
    //std::cout << "Sampling scalar at " << coord.coord[0] << ", " << coord.coord[1] << std::endl;
    GridCoord shiftedCoord = coord.shift();
    //std::cout << " | Shifted to " << shiftedCoord.coord[0] << ", " << shiftedCoord.coord[1] << std::endl;
    std::vector<int> flooredPos = std::vector<int>(n.size());
    std::vector<float> localCoord = std::vector<float>(n.size());
    for(int i = 0; i < n.size(); i++) {
        flooredPos.at(i) = std::floorf(shiftedCoord.coord[i]);
        localCoord.at(i) = shiftedCoord.coord[i] - float(flooredPos[i]);
    }
    //std::cout << " | Floored to " << flooredPos[0] << ", " << flooredPos[1] << std::endl;
    //std::cout << " | Local coord " << localCoord[0] << ", " << localCoord[1] << std::endl;

    std::vector<float> samplePoints;
    samplePoints.reserve(1 << n.size());
    for (int mask = 0; mask < (1 << n.size()); mask++) {
        std::vector<int> indices = flooredPos;
        for (int i = 0; i < n.size(); i++) {
            if (mask & (1 << i)) {
                indices[i]++;
            }
            indices[i] = std::clamp(indices[i], 0, n.at(i)-1);
        }

        //std::cout << " | Adding sample at " << indices[0] << ", " << indices[1] << std::endl;
        samplePoints.emplace_back(
            getExactScalar(type, MultiIndex(indices, scalarFields[type].getContext()))
        );
        //std::cout << " | Value: " << getExactScalar(type, MultiIndex(indices, scalarFields[type].getContext())) << std::endl;
    }

    return nLerp(
        samplePoints,
        localCoord,
        n.size()
    );
}

float Grid::getDivergence(MultiIndex idx) const {
    float div = 0;
    for(int i = 0; i < n.size(); i++) {
        auto ind = idx.getIndices();
        ind[i]++;
        MultiIndex succ(ind, velocities[i].getContext());
        MultiIndex prec(idx.getIndices(), velocities[i].getContext());
        div += (getExactVelocity(static_cast<Axis>(i), succ) - getExactVelocity(static_cast<Axis>(i), prec)) / d.at(i);
    }
    return div;
}

float Grid::getScalarGradient(ScalarFieldID type, MultiIndex idx, Axis axis) const {
    //std::cout << "Retreiving scalar gradient(" << idx.getIndices()[0] << ", " << idx.getIndices()[1] << ")" << std::endl;
    //std::cout << " | Left (" << idx.getPreceeding(axis).getIndices()[0] << ", " << idx.getPreceeding(axis).getIndices()[1] << ")" << std::endl;
    //std::cout << " | Right (" << idx.getIndices()[0] << ", " << idx.getIndices()[1] << ")" << std::endl;
    MultiIndex cellIndex(idx.getIndices(), cellType.getContext());
    return (getExactScalar(type, cellIndex) - getExactScalar(type, cellIndex.getPreceeding(axis))) / d.at(axis);
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