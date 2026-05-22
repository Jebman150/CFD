#pragma once

#include "queryBase.hpp"
#include "views.hpp"

namespace engine {

template<>
class Query<FaceView> {
public:
    using T = FaceView;

    Query(std::vector<T> _data)
     :  data(_data), 
        condition({[](T& t) { return true; }}) {}

    Query& withCondition(std::function<bool(T&)> cond) {
        condition.emplace_back(cond);
        return *this;
    }

    Query& normalTo(Axis axis) {
        condition.emplace_back([axis](T& face) {
            return face.axis == axis;
        });
        return *this;
    }

    Query& tangentTo(Axis axis) {
        condition.emplace_back([axis](T& face) {
            return face.axis != axis;
        });
        return *this;
    }

    Query& inPlane(Plane plane) {
        condition.emplace_back([plane](T& face) {
            return face.idx.getMultiIndex().at(static_cast<int>(plane.axis)) == plane.side;
        });
        return *this;
    }

    Query& isSolid() {
        condition.emplace_back([](T& face) {
            return face.type != FaceType::Fluid_Fluid;
        });
        return *this;
    }

    Query& notType(FaceType type) {
        condition.emplace_back([type](T& face) {
            return face.type != type;
        });
        return *this;
    }

    Query& isInternal() {
        condition.emplace_back([](T& face) {
            return face.type == FaceType::Fluid_Fluid;
        });
        return *this;
    }

    template<typename Func>
    bool execute(Func func) {
        for(T& t : data) {
            bool passed = true;
            for(auto& cond : condition) {
                if(!cond(t)) {
                    passed = false;
                    break;
                }
            }
            if(!passed) continue;
        }
        return true;
    }

    bool setVelocity(float val) {
        return execute([val] (T& face) {
            face.value = val;
        });
    }

    bool applyBoundaryCondition(float velocity) {
        return execute([&velocity] (T& face) {
            face.value = velocity;
            if(velocity < 1e-8) face.type = FaceType::Fluid_Solid;
        });
    }
private:
    std::vector<T> data;
    std::vector<std::function<bool(T&)>> condition;
};

template<>
class Query<ConstFaceView> {
public:
    using T = ConstFaceView;

    Query(std::vector<T> _data)
     :  data(_data), 
        condition({[](T& t) { return true; }}) {}

    Query& withCondition(std::function<bool(T&)> cond) {
        condition.emplace_back(cond);
        return *this;
    }

    Query& normalTo(Axis axis) {
        condition.emplace_back([axis](T& face) {
            return face.axis == axis;
        });
        return *this;
    }

    Query& tangentTo(Axis axis) {
        condition.emplace_back([axis](T& face) {
            return face.axis != axis;
        });
        return *this;
    }

    Query& inPlane(Plane plane) {
        condition.emplace_back([plane](T& face) {
            return face.idx.getMultiIndex().at(static_cast<int>(plane.axis)) == plane.side;
        });
        return *this;
    }

    Query& isSolid() {
        condition.emplace_back([](T& face) {
            return face.type != FaceType::Fluid_Fluid;
        });
        return *this;
    }

    Query& notType(FaceType type) {
        condition.emplace_back([type](T& face) {
            return face.type != type;
        });
        return *this;
    }

    Query& isInternal() {
        condition.emplace_back([](T& face) {
            return face.type == FaceType::Fluid_Fluid;
        });
        return *this;
    }

    template<typename Func>
    bool execute(Func func) {
        for(T& t : data) {
            bool passed = true;
            for(auto& cond : condition) {
                if(!cond(t)) {
                    passed = false;
                    break;
                }
            }
            if(!passed) continue;
        }
        return true;
    }
private:
    std::vector<T> data;
    std::vector<std::function<bool(T&)>> condition;
};

template<>
class Query<CellView> {
public:
    using T = CellView;

    Query(std::vector<T> _data)
     :  data(_data), 
        condition({[](T& t) { return true; }}) {}

    Query& withCondition(std::function<bool(T&)> cond) {
        condition.emplace_back(cond);
        return *this;
    }

    Query& inPlane(Plane plane) {
        condition.emplace_back([plane](T& cell) {
            return cell.idx.getMultiIndex().at(static_cast<int>(plane.axis)) == plane.side;
        });
        return *this;
    }

    Query& inSphere(float radius, GridCoord center) {
        condition.emplace_back([radius, center](T& cell) {
            return cell.coord.dist(center) < radius;
        });
        return *this;
    }

    Query& isSolid() {
        condition.emplace_back([](T& cell) {
            return cell.type == CellType::Solid;
        });
        return *this;
    }

    Query& isFluid() {
        condition.emplace_back([](T& cell) {
            return cell.type == CellType::Fluid;
        });
        return *this;
    }

    template<typename Func>
    bool execute(Func func) {
        for(T& t : data) {
            bool passed = true;
            for(auto& cond : condition) {
                if(!cond(t)) {
                    passed = false;
                    break;
                }
            }
            if(!passed) continue;
        }
        return true;
    }

    bool setSolid() {
        return execute([] (T& cell) {
            cell.type = CellType::Solid;
        });
    }
private:
    std::vector<T> data;
    std::vector<std::function<bool(T&)>> condition;
};

template<>
class Query<ConstCellView> {
public:
    using T = ConstCellView;

    Query(std::vector<T> _data)
     :  data(_data), 
        condition({[](T& t) { return true; }}) {}

    Query& withCondition(std::function<bool(T&)> cond) {
        condition.emplace_back(cond);
        return *this;
    }

    Query& inPlane(Plane plane) {
        condition.emplace_back([plane](T& cell) {
            return cell.idx.getMultiIndex().at(static_cast<int>(plane.axis)) == plane.side;
        });
        return *this;
    }

    Query& inSphere(float radius, GridCoord center) {
        condition.emplace_back([radius, center](T& cell) {
            return cell.coord.dist(center) < radius;
        });
        return *this;
    }

    Query& isSolid() {
        condition.emplace_back([](T& cell) {
            return cell.type == CellType::Solid;
        });
        return *this;
    }

    Query& isFluid() {
        condition.emplace_back([](T& cell) {
            return cell.type == CellType::Fluid;
        });
        return *this;
    }

    template<typename Func>
    bool execute(Func func) {
        for(T& t : data) {
            bool passed = true;
            for(auto& cond : condition) {
                if(!cond(t)) {
                    passed = false;
                    break;
                }
            }
            if(!passed) continue;
            func(t);
        }
        return true;
    }
private:
    std::vector<T> data;
    std::vector<std::function<bool(T&)>> condition;
};

}