#pragma once

#include <Eigen/Dense>
#include <vector>
#include <array>

#include "navigation.hpp"

namespace engine {

using namespace navigation;

struct GridCoord {
    float x, y, z;

    GridCoord() : x(0), y(0), z(0) {}
    GridCoord(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}
    GridCoord(Eigen::Vector3f v) : x(v.x()), y(v.y()), z(v.z()) {}

    operator Eigen::Vector3f()  {
        return {x, y, z};
    }

    GridCoord& operator=(const Eigen::Vector3f& v) {
        x = v.x();
        y = v.y();
        z = v.z();
        return *this;
    }
};

enum FaceType {
    Fluid_Fluid,
    Fluid_Solid,
    Solid_Fluid,
    Solid_Solid
};

struct Plane {
    Axis axis;
    int side;
};

struct FaceView {
    float& value;
    Index3D idx;
    GridCoord coord;
    FaceType& type;
    Axis axis;

    bool isInPlane(Plane plane) const {
        switch(plane.axis) {
            case Axis::X:
                return idx.i == plane.side;
            case Axis::Y:
                return idx.j == plane.side;
            case Axis::Z:
                return idx.k == plane.side;
        }
        return false;
    }
};

struct ConstFaceView {
    const float& value;
    Index3D idx;
    GridCoord coord;
    const FaceType& type;
    Axis axis;
};

enum CellType {
    Fluid,
    Solid,
    Empty
};

struct CellView {
    Index3D idx;
    GridCoord coord;
    CellType& type;
};

struct ConstCellView {
    Index3D idx;
    GridCoord coord;
    CellType type;
    float offset;
};

enum ScalarFieldID {
    Pressure,
    Smoke,
    NumFields
};

struct ScalarField {
    std::vector<float> data;
    bool advect;
};

class Grid {
    // Grid topology
    int nx, ny, nz;
    float dx, dy, dz;

    // Grid data
    std::vector<CellType> cellType;

    std::array<ScalarField, ScalarFieldID::NumFields> scalarFields;

    // Edge data
    std::vector<float> u;
    std::vector<float> v;
    std::vector<float> w;
    std::vector<FaceType> uType;
    std::vector<FaceType> vType;
    std::vector<FaceType> wType;

    float getExactU(Index3D idx) const;
    float getExactV(Index3D idx) const;
    float getExactW(Index3D idx) const;
    float getExactScalar(ScalarFieldID type, Index3D idx) const;

    void moveCoord(Direction dir, GridCoord& coord, float magnitude) const;

    float getScalarGradX(ScalarFieldID type, GridCoord coord) const;
    float getScalarGradY(ScalarFieldID type, GridCoord coord) const;
    float getScalarGradZ(ScalarFieldID type, GridCoord coord) const;

    inline int uIndex(Index3D idx) const {
        return idx.i + (nx+1) * (idx.j + ny * idx.k);
    }

    inline int vIndex(Index3D idx) const {
        return idx.i + nx * (idx.j + (ny+1) * idx.k);
    }

    inline int wIndex(Index3D idx) const {
        return idx.i + nx * (idx.j + ny * idx.k);
    }

    inline bool isValid(Index3D idx) const {
        return (idx.i >= 0 && idx.i < nx) &&
            (idx.j >= 0 && idx.j < ny) &&
            (idx.k >= 0 && idx.k < nz);
    }
public:
    void initialize(Eigen::Vector3i _gridSize, std::array<float, 6> _boundaryCondition);

    // Setter
    void setVelocityU(Index3D idx, float val) { u.at(uIndex(idx)) = val; }
    void setVelocityV(Index3D idx, float val) { v.at(vIndex(idx)) = val; }
    void setVelocityW(Index3D idx, float val) { w.at(wIndex(idx)) = val; }

    void setSolid(Index3D idx) { cellType.at(cellIndex(idx)) = CellType::Solid; }

    void setScalarField(ScalarFieldID type, Index3D idx, float val) { scalarFields[static_cast<int>(type)].data.at(cellIndex(idx)) = val; }
    void overrideScalarField(ScalarFieldID type, std::vector<float> data) { scalarFields[static_cast<int>(type)].data = data; }

    void overrideVelocities(std::array<std::vector<float>, Axis::Dim> buffer) {
        u = buffer[Axis::X];
        v = buffer[Axis::Y];
        w = buffer[Axis::Z];
    }

    // Computational getter
    Eigen::Vector3f interpolateVelocity(GridCoord coord) const;
    float getDivergence(GridCoord coord) const;
    float getScalarGradient(ScalarFieldID type, GridCoord coord, Axis axis) const;

    // Getter
    float getVelocityU(GridCoord coord) const;
    float getVelocityV(GridCoord coord) const;
    float getVelocityW(GridCoord coord) const;

    CellType getCellType(Index3D idx) const {
        if(!isValid(idx)) return CellType::Solid;
        return cellType.at(cellIndex(idx)); 
    }

    float getScalarField(ScalarFieldID type, GridCoord coord) const;
    ScalarField& getScalarField(ScalarFieldID type) { return scalarFields[type]; }

    float getMaxVelocity() const;

    inline int cellIndex(Index3D idx) const {
        return idx.i + nx * (idx.j + ny * idx.k);
    }

    int getVelocityIndex(Axis axis, Index3D idx) {
        switch(axis) {
            case Axis::X: return uIndex(idx);
            case Axis::Y: return vIndex(idx);
            case Axis::Z: return wIndex(idx);
        } return uIndex(idx);
    }

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

    Plane getBoundaryPlane(Direction dir) const {
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

    Axis getAxis(Direction dir) const {
        switch(dir) {
            case Left: return Axis::X;
            case Right: return Axis::X;
            case Top: return Axis::Y;
            case Bottom: return Axis::Y;
            case Front: return Axis::Z;
            case Back: return Axis::Z;
        }
        return Axis::X;
    }

    Eigen::Vector3f getBasisVector(Direction dir) const {
        switch(dir) {
            case Left: return {-1, 0, 0};
            case Right: return {1, 0, 0};
            case Top: return {0, -1, 0};
            case Bottom: return {0, 1, 0};
            case Front: return {0, -1, 0};
            case Back: return {0, 1, 0};
        }
        return {-1, 0, 0};
    }

    // Callback iterator
    template<typename Func>
    void forEachCell(Func func) { forEachCell(func, [](CellView cell) { return true; }); }

    template<typename Func, typename Condition>
    void forEachCell(Func func, Condition cond) {
        for(Indexer3D idxer(nx, ny, nz); !idxer.end(); idxer++) {
            Index3D idx = idxer.get();
            CellView view = {
                idx,
                coordFromCellIndex(idx),
                cellType.at(cellIndex(idx))
            };
            if(cond(view)) func(view);
        }
    }

    template<typename Func>
    void queryCells(Func func) const { queryCells(func, [](CellView cell) { return true; }); }

    template<typename Func, typename Condition>
    void queryCells(Func func, Condition cond) const {
        for(Indexer3D idxer(nx, ny, nz); !idxer.end(); idxer++) {
            Index3D idx = idxer.get();
            ConstCellView view = {
                idx,
                coordFromCellIndex(idx),
                cellType.at(cellIndex(idx)),
                1.f/(dx*dx)
            };
            if(cond(view)) func(view);
        }
    }

    template<typename Func>
    void forEachNeighbour(Index3D idx, Func func) const {
        auto neighbourIdx = idx.getNeighbours();
        for(auto nIdx : neighbourIdx) {
            if(!isValid(nIdx)) continue;
            ConstCellView view = {
                nIdx,
                coordFromCellIndex(nIdx),
                getCellType(nIdx),
                1.f/(dx * dx)
            };
            func(view);
        }
    }

    template<typename Func>
    void forEachCellFace(Index3D idx, Func func) {
        auto indices = idx.getCellFaceIndices();
        for(int i = 0; i < 2; i++) {
            FaceView view = {
                u.at(uIndex(indices[i])),
                indices[i],
                coordFromUIndex(indices[i]),
                uType.at(uIndex(indices[i])),
                Axis::X
            };
            func(view);
        }
        for(int i = 2; i < 4; i++) {
            FaceView view = {
                v.at(vIndex(indices[i])),
                indices[i],
                coordFromVIndex(indices[i]),
                vType.at(vIndex(indices[i])),
                Axis::Y
            };
            func(view);
        }
        for(int i = 4; i < 6; i++) {
            FaceView view = {
                w.at(wIndex(indices[i])),
                indices[i],
                coordFromWIndex(indices[i]),
                wType.at(wIndex(indices[i])),
                Axis::Z
            };
            func(view);
        }
    }

    // Face iterator
    template<typename Func>
    void forEachFace(Plane plane, Func func) {
        switch(plane.axis) {
            case Axis::X:
                forEachXFace(func, [&plane](FaceView face) {
                    return face.idx.i == plane.side;
                });
                return;
            case Axis::Y:
                forEachYFace(func, [&plane](FaceView face) {
                    return face.idx.j == plane.side;
                });
                return;
            case Axis::Z:
                forEachZFace(func, [&plane](FaceView face) {
                    return face.idx.k == plane.side;
                });
                return;
        }
    }

    template<typename Func>
    void forEachTangentFace(Plane plane, Func func) {
        switch(plane.axis) {
            case Axis::X:
                forEachYFace(func, [&plane](FaceView face) {
                    return face.idx.i == plane.side;
                });
                forEachZFace(func, [&plane](FaceView face) {
                    return face.idx.i == plane.side;
                });
                return;
            case Axis::Y:
                forEachXFace(func, [&plane](FaceView face) {
                    return face.idx.j == plane.side;
                });
                forEachZFace(func, [&plane](FaceView face) {
                    return face.idx.j == plane.side;
                });
                return;
            case Axis::Z:
                forEachXFace(func, [&plane](FaceView face) {
                    return face.idx.k == plane.side;
                });
                forEachYFace(func, [&plane](FaceView face) {
                    return face.idx.k == plane.side;
                });
                return;
        }
    }

    template<typename Func>
    void forEachFace(Func func) { forEachFace(func, [](FaceView face) { return true; }); }

    template<typename Condition, typename Func>
    void forEachFace(Func func, Condition cond) {
        forEachXFace(func, cond);
        forEachYFace(func, cond);
        forEachZFace(func, cond);
    }

    template<typename Func>
    void queryTangentFaces(Plane plane, Func func) const {
        switch(plane.axis) {
            case Axis::X:
                queryYFaces(func, [&plane](ConstFaceView face) {
                    return face.idx.i == plane.side;
                });
                queryZFaces(func, [&plane](ConstFaceView face) {
                    return face.idx.i == plane.side;
                });
                return;
            case Axis::Y:
                queryXFaces(func, [&plane](ConstFaceView face) {
                    return face.idx.j == plane.side;
                });
                queryZFaces(func, [&plane](ConstFaceView face) {
                    return face.idx.j == plane.side;
                });
                return;
            case Axis::Z:
                queryXFaces(func, [&plane](ConstFaceView face) {
                    return face.idx.k == plane.side;
                });
                queryYFaces(func, [&plane](ConstFaceView face) {
                    return face.idx.k == plane.side;
                });
                return;
        }
    }

    template<typename Func>
    void queryFaces(Func func) const { queryFaces(func, [](ConstFaceView face) { return true; }); }

    template<typename Condition, typename Func>
    void queryFaces(Func func, Condition cond) const {
        queryXFaces(func, cond);
        queryYFaces(func, cond);
        queryZFaces(func, cond);
    }

    template<typename Func>
    void forEachXFace(Func func) { forEachXFace(func, [](FaceView face) { return true; }); }

    template<typename Func, typename Condition>
    void forEachXFace(
        Func callback,
        Condition condition
    ) {
        for(Indexer3D idxer(nx+1, ny, nz); !idxer.end(); idxer++) {
            Index3D idx = idxer.get();
            FaceView face = {
                u.at(uIndex(idx)),
                idx,
                coordFromUIndex(idx),
                uType.at(uIndex(idx)),
                Axis::X
            };
            if(!condition(face)) continue;
            callback(face);
        }
    }

    template<typename Func, typename Condition>
    void queryXFaces(
        Func callback,
        Condition condition
    ) const {
        for(Indexer3D idxer(nx+1, ny, nz); !idxer.end(); idxer++) {
            Index3D idx = idxer.get();
            ConstFaceView face = {
                u.at(uIndex(idx)),
                idx,
                coordFromUIndex(idx),
                uType.at(uIndex(idx)),
                Axis::X
            };
            if(!condition(face)) continue;
            callback(face);
        }
    }

    template<typename Func>
    void forEachYFace(Func func) { forEachYFace(func, [](FaceView face) { return true; }); }

    template<typename Func, typename Condition>
    void forEachYFace(
        Func callback,
        Condition condition
    ) {
        for(Indexer3D idxer(nx, ny+1, nz); !idxer.end(); idxer++) {
            Index3D idx = idxer.get();
            FaceView face = {
                v.at(vIndex(idx)),
                idx,
                coordFromVIndex(idx),
                vType.at(vIndex(idx)),
                Axis::Y
            };
            if(!condition(face)) continue;
            callback(face);
        }
    }

    template<typename Func, typename Condition>
    void queryYFaces(
        Func callback,
        Condition condition
    ) const {
        for(Indexer3D idxer(nx, ny+1, nz); !idxer.end(); idxer++) {
            Index3D idx = idxer.get();
            ConstFaceView face = {
                v.at(vIndex(idx)),
                idx,
                coordFromVIndex(idx),
                vType.at(vIndex(idx)),
                Axis::Y
            };
            if(!condition(face)) continue;
            callback(face);
        }
    }

    template<typename Func>
    void forEachZFace(Func func) { forEachZFace(func, [](FaceView face) { return true; }); }

    template<typename Func, typename Condition>
    void forEachZFace(
        Func callback,
        Condition condition
    ) {
        for(Indexer3D idxer(nx, ny, nz+1); !idxer.end(); idxer++) {
            Index3D idx = idxer.get();
            FaceView face = {
                w.at(wIndex(idx)),
                idx,
                coordFromWIndex(idx),
                wType.at(wIndex(idx)),
                Axis::Z
            };
            if(!condition(face)) continue;
            callback(face);
        }
    }

    template<typename Func, typename Condition>
    void queryZFaces(
        Func callback,
        Condition condition
    ) const{
        for(Indexer3D idxer(nx, ny, nz+1); !idxer.end(); idxer++) {
            Index3D idx = idxer.get();
            ConstFaceView face = {
                w.at(wIndex(idx)),
                idx,
                coordFromWIndex(idx),
                wType.at(wIndex(idx)),
                Axis::Z
            };
            if(!condition(face)) continue;
            callback(face);
        }
    }
};

}