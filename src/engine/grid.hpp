#pragma once

#include <Eigen/Dense>
#include <vector>
#include <array>

struct GridCoord {
    float x, y, z;
};

struct Index3D {
    int i, j, k;
};

struct ScalarField {
    std::vector<float> data;
    bool advect;
};

enum Axis {
    X, Y, Z, Dim
};

struct FaceView {
    float& value;
    Index3D idx;
    GridCoord coord;
    Axis axis;
};

struct ConstFaceView {
    float value;
    Index3D idx;
    GridCoord coord;
    Axis axis;
};

struct Plane {
    Axis axis;
    int side;
};

enum CellType {
    Fluid,
    Solid,
    Empty
};

enum ScalarFieldID {
    Pressure,
    NumFields
};

enum BoundaryCondition {
    Wall,
    Open,
    Inflow,
    Outflow,
    partialInflow,
    partialOutflow
};

enum Direction {
    Left,
    Right,
    Top,
    Bottom,
    Front,
    Back,
    NUM
};

class Grid {
    // Grid topology
    int nx, ny, nz;
    float dx, dy, dz;

    std::array<BoundaryCondition, 6> boundaryCondition;

    // Grid data
    std::vector<CellType> cellType;

    std::array<std::vector<float>, ScalarFieldID::NumFields> scalarFields;

    // Edge data
    std::vector<float> u;
    std::vector<float> v;
    std::vector<float> w;

    float getExactU(Index3D idx) const;
    float getExactV(Index3D idx) const;
    float getExactW(Index3D idx) const;
    float getExactScalar(ScalarFieldID type, Index3D idx) const;

    float getScalarGradX(ScalarFieldID type, GridCoord coord) const;
    float getScalarGradY(ScalarFieldID type, GridCoord coord) const;
    float getScalarGradZ(ScalarFieldID type, GridCoord coord) const;

    inline int cellIndex(Index3D idx) const {
        return idx.i + nx * (idx.j + ny * idx.k);
    }

    inline int uIndex(Index3D idx) const {
        return idx.i + (nx+1) * (idx.j + ny * idx.k);
    }

    inline int vIndex(Index3D idx) const {
        return idx.i + nx * (idx.j + (ny+1) * idx.k);
    }

    inline int wIndex(Index3D idx) const {
        return idx.i + nx * (idx.j + ny * idx.k);
    }
public:
    void initialize(Eigen::Vector3i _gridSize, std::array<BoundaryCondition, 6> _boundaryCondition);

    // Setter
    void setVelocityU(Index3D idx, float val) { u.at(uIndex(idx)) = val; }
    void setVelocityV(Index3D idx, float val) { v.at(vIndex(idx)) = val; }
    void setVelocityW(Index3D idx, float val) { w.at(wIndex(idx)) = val; }

    void setScalarField(ScalarFieldID type, Index3D idx, float val) { scalarFields[static_cast<int>(type)].at(cellIndex(idx)) = val; }

    void overrideVelocities(std::array<std::vector<float>, Axis::Dim> buffer) {
        u = buffer[Axis::X];
        v = buffer[Axis::Y];
        w = buffer[Axis::Z];
    }

    // Computational getter
    Eigen::Vector3d interpolateVelocity(GridCoord coord) const;
    float getDivergence(GridCoord coord) const;
    float getScalarGradient(ScalarFieldID type, GridCoord coord, Axis axis) const;

    // Getter
    float getVelocityU(GridCoord coord) const;
    float getVelocityV(GridCoord coord) const;
    float getVelocityW(GridCoord coord) const;

    float getScalarField(ScalarFieldID type, GridCoord coord) const;

    float getMaxVelocity() const;

    int getVelocityIndex(Axis axis, Index3D idx) {
        switch(axis) {
            case Axis::X: return uIndex(idx);
            case Axis::Y: return vIndex(idx);
            case Axis::Z: return wIndex(idx);
        } return uIndex(idx);
    }

    BoundaryCondition getBoundaryCondition(Direction direction) const { return boundaryCondition.at(direction); }

    int getWidth() const { return nx; }
    int getHeight() const { return ny; }
    int getDepth() const { return nz; }
    int getSize(Axis axis) const {
        switch(axis) {
            case Axis::X: return nx;
            case Axis::Y: return ny;
            case Axis::Z: return nz;
        } return nx;
    }
    Eigen::Vector2i getDimensions(Axis axis) const {
        switch(axis) {
            case Axis::X: return {nz, ny};
            case Axis::Y: return {nx, nz};
            case Axis::Z: return {nx, ny};
        } return {nx, ny};
    }

    float getDx() const { return dx; }
    float getDy() const { return dy; }
    float getDz() const { return dz; }

    GridCoord coordFromCellIndex(Index3D idx) const { return {idx.i + 0.5f, idx.j + 0.5f, idx.k + 0.5f}; }
    GridCoord coordFromUIndex(Index3D idx) const { return {float(idx.i), idx.j + 0.5f, idx.k + 0.5f}; }
    GridCoord coordFromVIndex(Index3D idx) const { return {idx.i + 0.5f, float(idx.j), idx.k + 0.5f}; }
    GridCoord coordFromWIndex(Index3D idx) const { return {idx.i + 0.5f, idx.j + 0.5f, float(idx.k)}; }

    Plane getBoundaryPlane(Direction dir) {
        switch(dir) {
            case Left: return {Axis::X, 0};
            case Right: return {Axis::X, nx};
            case Top: return {Axis::Y, 0};
            case Bottom: return {Axis::Y, ny};
            case Front: return {Axis::Z, 0};
            case Back: return {Axis::Z, nz};
        }
        return {Axis::X, 0};
    }

    // Callback iterator
    template<typename Func>
    void forEachVelocity(Plane plane, Func func) {
        switch(plane.axis) {
            case Axis::X:
                forEachXVelocity(plane.side, func);
                return;
            case Axis::Y:
                forEachYVelocity(plane.side, func);
                return;
            case Axis::Z:
                forEachZVelocity(plane.side, func);
                return;
        }
    }

    template<typename Func>
    void forEachTangentVelocity(Plane plane, Func func) const {
        switch(plane.axis) {
            case Axis::X:
                forEachTangentYVelocity(plane, func);
                forEachTangentZVelocity(plane, func);
                return;
            case Axis::Y:
                forEachTangentXVelocity(plane, func);
                forEachTangentZVelocity(plane, func);
                return;
            case Axis::Z:
                forEachTangentXVelocity(plane, func);
                forEachTangentYVelocity(plane, func);
                return;
        }
    }

    template<typename Func>
    void forEachXVelocity(int planeIdx, Func callback) {
        for(int j = 0; j < ny; j++) {
            for(int k = 0; k < nz; k++) {
                Index3D idx = {planeIdx, j, k};
                FaceView face = {
                    u.at(uIndex(idx)),
                    idx,
                    coordFromUIndex(idx),
                    Axis::X
                };
                callback(face);
            }
        }
    }

    template<typename Func>
    void forEachTangentXVelocity(Plane plane, Func callback) const {
        int secondDim = plane.axis == Axis::Y ? nz : ny;
        for(int i = 0; i < nx; i++) {
            for(int j = 0; j < secondDim; j++) {
                Index3D idx = plane.axis == Axis::Y ? Index3D{i, plane.side, j} : Index3D{i, j, plane.side};
                ConstFaceView face = {
                    u.at(uIndex(idx)),
                    idx,
                    coordFromUIndex(idx),
                    Axis::X
                };
                callback(face);
            }
        }
    }

    template<typename Func>
    void forEachYVelocity(int planeIdx, Func callback) {
        for(int i = 0; i < nx; i++) {
            for(int k = 0; k < nz; k++) {
                Index3D idx = {i, planeIdx, k};
                FaceView face = {
                    v.at(vIndex(idx)),
                    idx,
                    coordFromVIndex(idx),
                    Axis::Y
                };
                callback(face);
            }
        }
    }

    template<typename Func>
    void forEachTangentYVelocity(Plane plane, Func callback) const {
        int secondDim = plane.axis == Axis::X ? nz : nx;
        for(int i = 0; i < ny; i++) {
            for(int j = 0; j < secondDim; j++) {
                Index3D idx = plane.axis == Axis::X ? Index3D{plane.side, i, j} : Index3D{j, i, plane.side};
                ConstFaceView face = {
                    v.at(vIndex(idx)),
                    idx,
                    coordFromVIndex(idx),
                    Axis::Y
                };
                callback(face);
            }
        }
    }

    template<typename Func>
    void forEachZVelocity(int planeIdx, Func callback) {
        for(int i = 0; i < nx; i++) {
            for(int j = 0; j < ny; j++) {
                Index3D idx = {i, j, planeIdx};
                FaceView face = {
                    w.at(wIndex(idx)),
                    idx,
                    coordFromWIndex(idx),
                    Axis::Z
                };
                callback(face);
            }
        }
    }

    template<typename Func>
    void forEachTangentZVelocity(Plane plane, Func callback) const {
        int secondDim = plane.axis == Axis::Y ? nx : ny;
        for(int i = 0; i < nz; i++) {
            for(int j = 0; j < secondDim; j++) {
                Index3D idx = plane.axis == Axis::Y ? Index3D{j, plane.side, i} : Index3D{plane.side, j, i};
                ConstFaceView face = {
                    w.at(wIndex(idx)),
                    idx,
                    coordFromWIndex(idx),
                    Axis::Z
                };
                callback(face);
            }
        }
    }
};