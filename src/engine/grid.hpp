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

struct ScalarField {
    std::vector<float> data;
    bool advect;
};

class Grid {
    // Grid topology
    int totalCellCount = 0;
    std::array<int, Axis::Dim> n;
    std::array<float, Axis::Dim> d;

    // Grid data
    std::vector<CellType> cellType;

    std::array<ScalarField, ScalarFieldID::NumFields> scalarFields;

    // Edge data
    std::array<std::vector<float>, Axis::Dim> velocities;
    std::array<std::vector<FaceType>, Axis::Dim> faceTypes;

    float getExactVelocity(Axis axis, MultiIndex idx) const;
    float getExactScalar(ScalarFieldID type, MultiIndex idx) const;

    float getScalarGradX(ScalarFieldID type, GridCoord coord) const;
    float getScalarGradY(ScalarFieldID type, GridCoord coord) const;
    float getScalarGradZ(ScalarFieldID type, GridCoord coord) const;
public:
    void initialize(std::array<int, Axis::Dim> _n, std::array<float, 6> _boundaryCondition);

    // Setter
    void setVelocity(Axis axis, MultiIndex idx, float val) { velocities[axis].at(idx.get()) = val; }

    void setSolid(MultiIndex idx) { cellType.at(idx.get()) = CellType::Solid; }

    void setScalarField(ScalarFieldID type, MultiIndex idx, float val) { scalarFields[static_cast<int>(type)].data.at(idx.get()) = val; }
    void overrideScalarField(ScalarFieldID type, std::vector<float> data) { scalarFields[static_cast<int>(type)].data = data; }

    void overrideVelocities(std::array<std::vector<float>, Axis::Dim> buffer) {
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
        return cellType.at(idx.get()); 
    }

    float getScalarField(ScalarFieldID type, GridCoord coord) const;
    ScalarField& getScalarField(ScalarFieldID type) { return scalarFields[type]; }

    float getMaxVelocity() const;

    int getCellCount() const { return totalCellCount; }
    int getSize(Axis axis) const { return n.at(axis); }
    std::array<int, Axis::Dim> getSize() const { return n; }
    int getMACCellCount(Axis axis) const { 
        auto nMac = getMACSize(axis);
        int sum = 0;
        for(int i = 0; i < Axis::Dim; i++) {
            sum *= nMac[i];
        }
        return sum;
    }
    std::array<int, Axis::Dim> getMACSize(Axis axis) const {
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
        for(MultiIndex idx(n); !idx.overflow(); idx++) {
            CellView view = {
                idx,
                GridCoord(idx),
                cellType.at(idx.get())
            };
            if(cond(view)) func(view);
        }
    }

    template<typename Func>
    void queryCells(Func func) const { queryCells(func, [](CellView cell) { return true; }); }

    template<typename Func, typename Condition>
    void queryCells(Func func, Condition cond) const {
        for(MultiIndex idx(n); !idx.overflow(); idx++) {
            ConstCellView view = {
                idx,
                GridCoord(idx),
                cellType.at(idx.get()),
                1.f/(d[0] * d[0])
            };
            if(cond(view)) func(view);
        }
    }

    template<typename Func>
    void forEachNeighbour(MultiIndex idx, Func func) const {
        auto neighbourIdx = idx.getPreceding();
        for(auto i : idx.getSucceeding()) neighbourIdx.emplace_back(i);

        for(auto nIdx : neighbourIdx) {
            if(!isValid(nIdx)) continue;
            ConstCellView view = {
                nIdx,
                GridCoord(nIdx),
                cellType.at(idx.get()),
                1.f/(d[0] * d[0])
            };
            func(view);
        }
    }

    // Face iterator
    template<typename Func>
    void forEachFace(Plane plane, Func func) {
        for(MultiIndex idx(getMACSize(plane.axis)); !idx.overflow(); idx++) {
            if(idx.getIndices()[plane.axis] == plane.side) continue;
            FaceView view = {
                velocities[plane.axis].at(idx.get()),
                idx,
                GridCoord(idx),
                faceTypes.at(idx.get()),
                plane.axis
            }
            func(view);
        }
    }

    template<typename Func>
    void forEachTangentFace(Plane plane, Func func) {
        for(int i = 0; i < Axis::Dim; i++) {
            for(MultiIndex idx(getMACSize(i)); !idx.overflow(); idx++) {
                if(idx.getIndices()[plane.axis] == plane.side && plane.axis != i) continue;
                FaceView view = {
                    velocities[i].at(idx.get()),
                    idx,
                    GridCoord(idx),
                    faceTypes.at(idx.get()),
                    static_cast<Axis>(i)
                }
                if(!cond(view)) continue;
                func(view);
            }
        }
    }

    template<typename Func>
    void forEachFace(Func func) { forEachFace(func, [](FaceView face) { return true; }); }

    template<typename Condition, typename Func>
    void forEachFace(Func func, Condition cond) {
        for(int i = 0; i < Axis::Dim; i++) {
            for(MultiIndex idx(getMACSize(i)); !idx.overflow(); idx++) {
                FaceView view = {
                    velocities[i].at(idx.get()),
                    idx,
                    GridCoord(idx),
                    faceTypes.at(idx.get()),
                    static_cast<Axis>(i)
                }
                if(!cond(view)) continue;
                func(view);
            }
        }
    }

    template<typename Func>
    void queryTangentFace(Plane plane, Func func) {
        for(int i = 0; i < Axis::Dim; i++) {
            for(MultiIndex idx(getMACSize(static_cast<Axis>(i))); !idx.overflow(); idx++) {
                if(idx.getIndices()[plane.axis] == plane.side && plane.axis != i) continue;
                ConstFaceView view = {
                    velocities[i].at(idx.get()),
                    idx,
                    GridCoord(idx),
                    faceTypes.at(idx.get()),
                    static_cast<Axis>(i)
                }
                func(view);
            }
        }
    }

    template<typename Func>
    void queryFaces(Func func) const { queryFaces(func, [](ConstFaceView face) { return true; }); }

    template<typename Condition, typename Func>
    void queryFaces(Func func, Condition cond) const {
        for(int i = 0; i < Axis::Dim; i++) {
            for(MultiIndex idx(getMACSize(static_cast<Axis>(i))); !idx.overflow(); idx++) {
                ConstFaceView view = {
                    velocities[i].at(idx.get()),
                    idx,
                    GridCoord(idx),
                    faceTypes.at(idx.get()),
                    static_cast<Axis>(i)
                }
                if(!cond(view)) continue;
                func(view);
            }
        }
    }

    void updateFaceTypes() {
        for(int i = 0; i < Axis::Dim; i++) {
            for(MultiIndex idx(getMACSize(static_cast<Axis>(i))); !idx.overflow(); idx++) {
                CellType left;
                CellType right;

                auto indices = idx.getIndices();
                indices[i] -= 1;
                if(indices[i] < 0) left = CellType::Solid;
                else cellType.at(MultiIndex(n, indices).get());


                indices[i] += 2;
                if(indices[i] >= n.at(i)) left = CellType::Solid;
                else cellType.at(MultiIndex(n, indices).get());
                
            }
        }
    }
};

}