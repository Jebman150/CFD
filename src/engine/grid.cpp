#include "grid.hpp"

#include <cmath>
#include <iostream>

/*
    Initializes the grid with random values

    @param gridSize
        2D-Vector for width and height of the discretization grid
    
*/
void Grid::initialize(Eigen::Vector2i _gridSize) {
    width = _gridSize.x();
    height = _gridSize.y();
    u = Eigen::MatrixXd::Zero(width + 1, height);
    v = Eigen::MatrixXd::Zero(width, height + 1);
    uBuffer = Eigen::MatrixXd::Zero(width + 1, height);
    vBuffer = Eigen::MatrixXd::Zero(width, height + 1);
    p = Eigen::MatrixXd::Random(width, height);
    deltaX = 1.0/double(width);
    deltaY = 1.0/double(height);
}

void Grid::updateU(Index2D idx, double val) {
    u(idx.i, idx.j) += val;
}

void Grid::updateV(Index2D idx, double val) {
    v(idx.i, idx.j) += val;
}

void Grid::setPressure(Index2D idx, double val) {
    p(idx.i, idx.j) = val;
}

void Grid::setVelocityU(Index2D idx, double val) {
    u(idx.i, idx.j) = val;
}

void Grid::setVelocityV(Index2D idx, double val) {
    v(idx.i, idx.j) = val;
}

void Grid::bufferVelocityU(Index2D idx, double val) {
    uBuffer(idx.i, idx.j) = val;
}

void Grid::bufferVelocityV(Index2D idx, double val) {
    vBuffer(idx.i, idx.j) = val;
}

void Grid::applyBuffer() {
    u = uBuffer;
    v = vBuffer;
}

void Grid::flushBuffer() {
    uBuffer = u;
    vBuffer = v;
}

float lerp(float x, float y, float t) {
    return x + (y-x) * t;
}

float biLerp(float ul, float ur, float bl, float br, Eigen::Vector2d pos) {
    float uInt = lerp(ul, ur, pos.x());
    float bInt = lerp(bl, br, pos.x());
    return lerp(uInt, bInt, pos.y());
}

/*
    Returns the velocity of an arbitrary position in the volume.
*/
Eigen::Vector2d Grid::interpolateVelocity(GridCoord position) const {
    float interpolatedX = getVelocityU(position);
    float interpolatedY = getVelocityV(position);
    return {interpolatedX, interpolatedY};
}

double Grid::getExactP(int x, int y) const {
    x = std::clamp(x, 0, width-1);
    y = std::clamp(y, 0, height-1);
    return p(x, y);
}

double Grid::getExactU(int x, int y) const {
    x = std::clamp(x, 0, width);
    y = std::clamp(y, 0, height-1);
    return u(x, y);
}

double Grid::getExactV(int x, int y) const {
    x = std::clamp(x, 0, width-1);
    y = std::clamp(y, 0, height);
    return v(x, y);
}

double Grid::getPressure(GridCoord coord) const {
    GridCoord shiftedCoord = {coord.x - 0.5, coord.y - 0.5};
    int x = std::floor(shiftedCoord.x);
    int y = std::floor(shiftedCoord.y);

    Eigen::Vector2d localCoord = {shiftedCoord.x - double(x), shiftedCoord.y - double(y)};

    double ul = getExactP(x, y);
    double ur = getExactP(x + 1, y);
    double bl = getExactP(x, y + 1);
    double br = getExactP(x + 1, y + 1);
    return biLerp(ul, ur, bl, br, localCoord);
}

double Grid::getVelocityU(GridCoord coord) const {
    GridCoord shiftedCoord = {coord.x, coord.y - 0.5};
    int x = std::floor(shiftedCoord.x);
    int y = std::floor(shiftedCoord.y);

    Eigen::Vector2d localCoord = {shiftedCoord.x - double(x), shiftedCoord.y - double(y)};

    double ul = getExactU(x, y);
    double ur = getExactU(x + 1, y);
    double bl = getExactU(x, y + 1);
    double br = getExactU(x+ 1, y + 1);
    return biLerp(ul, ur, bl, br, localCoord);
}

double Grid::getVelocityV(GridCoord coord) const {
    GridCoord shiftedCoord = {coord.x - 0.5, coord.y};
    int x = std::floor(shiftedCoord.x);
    int y = std::floor(shiftedCoord.y);

    Eigen::Vector2d localCoord = {shiftedCoord.x - double(x), shiftedCoord.y - double(y)};

    double ul = getExactV(x, y);
    double ur = getExactV(x + 1, y);
    double bl = getExactV(x, y + 1);
    double br = getExactV(x+ 1, y + 1);
    return biLerp(ul, ur, bl, br, localCoord);
}

double Grid::getDivergence(GridCoord coord) const {
    float dx = (getVelocityU({coord.x + 0.5, coord.y}) - getVelocityU({coord.x - 0.5, coord.y})) / deltaX;
    float dy = (getVelocityV({coord.x, coord.y + 0.5}) - getVelocityV({coord.x, coord.y - 0.5})) / deltaY;
    return dx + dy;
}