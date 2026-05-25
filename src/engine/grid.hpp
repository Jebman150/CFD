#pragma once

#include <Eigen/Dense>
#include <vector>
#include <array>

#include "navigation.hpp"

namespace engine {

using namespace navigation;

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
    MultiIndex idx;
    GridCoord coord;
    FaceType& type;
    Axis axis;

    bool isInPlane(Plane plane) const {
        return idx.getIndices()[plane.axis] == plane.side;
    }
};

struct ConstFaceView {
    const float& value;
    MultiIndex idx;
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
    MultiIndex idx;
    GridCoord coord;
    CellType& type;
};

struct ConstCellView {
    MultiIndex idx;
    GridCoord coord;
    CellType type;
    float offset;
};

enum ScalarFieldID {
    Pressure,
    Smoke,
    NumFields
};

template<typename T>
class Field {
private:
    IndexContext indexContext;
    std::vector<T> data;

public:
    bool advect;

    Field() : Field({0}) {}
    Field(std::vector<int> size): indexContext(size), data(std::vector<T>(indexContext.nMax)), advect(false) {}
    Field(std::vector<int> size, T _init): indexContext(size), data(std::vector<T>(indexContext.nMax, _init)), advect(false) {}

    T at(MultiIndex idx) const {
        assert(idx.checkContext(&indexContext));
        assert(idx.isValid());
        return data.at(idx.get());
    }

    T& at(MultiIndex idx) {
        assert(idx.checkContext(&indexContext));
        assert(idx.isValid());
        return data.at(idx.get());
    }

    MultiIndex begin() const { return MultiIndex(&indexContext); }
    MultiIndex end() const { return MultiIndex(indexContext.nMax, &indexContext); }
    void setData(const std::vector<T>& _data) { data = _data; }
    const IndexContext* getContext() const { return &indexContext; }

    const std::vector<T>& getData() const { return data; }
};

class Grid {
    // Grid topology
    int totalCellCount = 0;
    std::vector<int> n;
    std::vector<float> d;

    // Grid data
    Field<CellType> cellType;

    std::array<Field<float>, ScalarFieldID::NumFields> scalarFields;

    // Edge data
    std::array<Field<float>, Axis::Dim> velocities;
    std::array<Field<FaceType>, Axis::Dim> faceTypes;

    float getExactVelocity(Axis axis, MultiIndex idx) const;
    float getExactScalar(ScalarFieldID type, MultiIndex idx) const;

    float getScalarGradX(ScalarFieldID type, GridCoord coord) const;
    float getScalarGradY(ScalarFieldID type, GridCoord coord) const;
    float getScalarGradZ(ScalarFieldID type, GridCoord coord) const;
public:
    void initialize(std::vector<int> _n, std::array<float, 6> _boundaryCondition);

    // Setter
    void setVelocity(Axis axis, MultiIndex idx, float val) { velocities[axis].at(idx) = val; }

    void setSolid(MultiIndex idx) { cellType.at(idx) = CellType::Solid; }

    void setScalarField(ScalarFieldID type, MultiIndex idx, float val) { scalarFields[static_cast<int>(type)].at(idx) = val; }
    void overrideScalarField(ScalarFieldID type, std::vector<float> data) { scalarFields[static_cast<int>(type)].setData(data); }

    void overrideVelocities(std::array<Field<float>, Axis::Dim> buffer) {
        velocities = buffer;
    }

    // Computational getter
    Eigen::VectorXf interpolateVelocity(GridCoord coord) const;
    float getDivergence(MultiIndex idx) const;
    //float getScalarGradient(ScalarFieldID type, GridCoord coord, Axis axis) const;
    float getScalarGradient(ScalarFieldID type, MultiIndex idx, Axis axis) const;

    // Getter
    float getVelocity(Axis axis, GridCoord coord) const;

    CellType getCellType(MultiIndex idx) const {
        if(!idx.isValid()) return CellType::Solid;
        return cellType.at(idx); 
    }
    
    const Field<CellType>& getCellTypeField() const {
        return cellType; 
    }

    float getScalarField(ScalarFieldID type, GridCoord coord) const;
    Field<float>& getScalarField(ScalarFieldID type) { return scalarFields[type]; }

    float getMaxVelocity() const;

    const IndexContext* getCellContext() const { return cellType.getContext(); }
    MultiIndex getCellIndex() const { return cellType.begin(); }
    int getCellCount() const { return totalCellCount; }
    int getSize(Axis axis) const { return n.at(axis); }
    std::vector<int> getSize() const { return n; }
    int getMACCellCount(Axis axis) const { 
        auto nMac = getMACSize(axis);
        int sum = 1;
        for(int i = 0; i < Axis::Dim; i++) {
            sum *= nMac[i];
        }
        return sum;
    }
    std::vector<int> getMACSize(Axis axis) const {
        auto nMAC = n;
        nMAC[axis]++;
        return nMAC;
    }

    float getDx(Axis axis) const { return d.at(axis); }

    Plane getBoundaryPlane(Axis axis, bool end) const {
        return {axis, end ? n.at(axis) : 0};
    }

    // Callback iterator
    template<typename Func>
    void forEachCell(Func func) { forEachCell(func, [](CellView cell) { return true; }); }

    template<typename Func, typename Condition>
    void forEachCell(Func func, Condition cond) {
        for(MultiIndex idx = cellType.begin(); !idx.overflow(); idx++) {
            CellView view = {
                idx,
                GridCoord(idx),
                cellType.at(idx)
            };
            if(cond(view)) func(view);
        }
    }

    template<typename Func>
    void queryCells(Func func) const { queryCells(func, [](ConstCellView cell) { return true; }); }

    template<typename Func, typename Condition>
    void queryCells(Func func, Condition cond) const {
        for(MultiIndex idx = cellType.begin(); !idx.overflow(); idx++) {
            ConstCellView view = {
                idx,
                GridCoord(idx),
                cellType.at(idx),
                1.f/(d[0] * d[0])
            };
            if(cond(view)) func(view);
        }
    }

    template<typename Func>
    void queryCells(Plane plane, Func func) const {
        for(MultiIndex idx = cellType.begin(); !idx.overflow(); idx++) {
            if(idx.getIndices()[plane.axis] != plane.side) continue;
            ConstCellView view = {
                idx,
                GridCoord(idx),
                cellType.at(idx),
                1.f/(d[0] * d[0])
            };
            func(view);
        }
    }

    template<typename Func>
    void forEachNeighbour(MultiIndex idx, Func func) const {
        forEachSuccessor(idx, func);
        forEachPredecessor(idx, func);
    }

    template<typename Func>
    void forEachPredecessor(MultiIndex idx, Func func) const {
        auto neighbourIdx = idx.getPreceeding();
        for(auto nIdx : neighbourIdx) {
            if(!nIdx.isValid()) continue;
            ConstCellView view = {
                nIdx,
                GridCoord(nIdx),
                cellType.at(nIdx),
                1.f/(d[0] * d[0])
            };
            func(view);
        }
    }

    template<typename Func>
    void forEachSuccessor(MultiIndex idx, Func func) const {
        auto neighbourIdx = idx.getSucceeding();
        for(auto nIdx : neighbourIdx) {
            if(!nIdx.isValid()) continue;
            ConstCellView view = {
                nIdx,
                GridCoord(nIdx),
                cellType.at(nIdx),
                1.f/(d[0] * d[0])
            };
            func(view);
        }
    }

    // Face iterator
    template<typename Func>
    void forEachFace(Plane plane, Func func) {
        for(MultiIndex idx = velocities[plane.axis].begin(); !idx.overflow(); idx++) {
            if(idx.getIndices()[plane.axis] != plane.side) continue;
            FaceView view = {
                velocities[plane.axis].at(idx),
                idx,
                    GridCoord(idx, static_cast<Axis>(plane.axis)),
                faceTypes[plane.axis].at(idx),
                plane.axis
            };
            func(view);
        }
    }

    template<typename Func>
    void forEachTangentFace(Plane plane, Func func) {
        for(int i = 0; i < Axis::Dim; i++) {
            for(MultiIndex idx = velocities[i].begin(); !idx.overflow(); idx++) {
                if(idx.getIndices()[plane.axis] != plane.side && plane.axis != i) continue;
                FaceView view = {
                    velocities[i].at(idx),
                    idx,
                    GridCoord(idx, static_cast<Axis>(i)),
                    faceTypes[i].at(idx),
                    static_cast<Axis>(i)
                };
                func(view);
            }
        }
    }

    template<typename Func>
    void forEachFace(Func func) { forEachFace(func, [](FaceView face) { return true; }); }

    template<typename Condition, typename Func>
    void forEachFace(Func func, Condition cond) {
        for(int i = 0; i < Axis::Dim; i++) {
            for(MultiIndex idx = velocities[i].begin(); !idx.overflow(); idx++) {
                FaceView view = {
                    velocities[i].at(idx),
                    idx,
                    GridCoord(idx, static_cast<Axis>(i)),
                    faceTypes[i].at(idx),
                    static_cast<Axis>(i)
                };
                if(!cond(view)) continue;
                func(view);
            }
        }
    }

    template<typename Func>
    void queryFaces(Plane plane, Func func) {
        for(MultiIndex idx = velocities[plane.axis].begin(); !idx.overflow(); idx++) {
            if(idx.getIndices()[plane.axis] != plane.side) continue;
            ConstFaceView view = {
                velocities[plane.axis].at(idx),
                idx,
                GridCoord(idx, static_cast<Axis>(plane.axis)),
                faceTypes[plane.axis].at(idx),
                plane.axis
            };
            func(view);
        }
    }

    template<typename Func>
    void queryTangentFaces(Plane plane, Func func) const {
        for(int i = 0; i < Axis::Dim; i++) {
            for(MultiIndex idx = velocities[plane.axis].begin(); !idx.overflow(); idx++) {
                if(idx.getIndices()[plane.axis] != plane.side && plane.axis != i) continue;
                ConstFaceView view = {
                    velocities[i].at(idx),
                    idx,
                    GridCoord(idx, static_cast<Axis>(i)),
                    faceTypes[i].at(idx),
                    static_cast<Axis>(i)
                };
                func(view);
            }
        }
    }

    template<typename Func>
    void queryFaces(Func func) const { queryFaces(func, [](ConstFaceView face) { return true; }); }

    template<typename Condition, typename Func>
    void queryFaces(Func func, Condition cond) const {
        for(int i = 0; i < Axis::Dim; i++) {
            for(MultiIndex idx = velocities[i].begin(); !idx.overflow(); idx++) {
                ConstFaceView view = {
                    velocities[i].at(idx),
                    idx,
                    GridCoord(idx, static_cast<Axis>(i)),
                    faceTypes[i].at(idx),
                    static_cast<Axis>(i)
                };
                if(!cond(view)) continue;
                func(view);
            }
        }
    }

    void updateFaceTypes() {
        for(int i = 0; i < Axis::Dim; i++) {
            for(MultiIndex idx = faceTypes[i].begin(); !idx.overflow(); idx++) {
                CellType left;
                CellType right;

                auto cellIdx = idx.transformedToContext(cellType.getContext());
                if(!cellIdx.isValid()) right = CellType::Solid;
                else right = cellType.at(cellIdx);

                cellIdx.advance(static_cast<Axis>(i), -1);
                if(!cellIdx.isValid()) left = CellType::Solid;
                else left = cellType.at(cellIdx);
                
                if(left == CellType::Solid) {
                    if(right == CellType::Solid) {
                        faceTypes[i].at(idx) = FaceType::Solid_Solid;
                    } else {
                        faceTypes[i].at(idx) = FaceType::Solid_Fluid;
                    }
                } else {
                    if(right == CellType::Solid) {
                        faceTypes[i].at(idx) = FaceType::Fluid_Solid;
                    } else {
                        faceTypes[i].at(idx) = FaceType::Fluid_Fluid;
                    }
                }
            }
        }
    }
};

}