#pragma once

#include "navigation.hpp"
#include <functional>
#include <vector>

namespace engine {

using namespace navigation;

template<typename T>
class Query {
public:
    Query(std::vector<T> _data)
     :  data(_data), 
        condition({[](T& t) { return true; }}) {}

    Query& withCondition(std::function<bool(T&)> cond) {
        condition.emplace_back(cond);
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
    Grid& grid;
    std::vector<std::function<bool(T&)>> condition;
};

}