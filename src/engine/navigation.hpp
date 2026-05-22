#pragma once

#include <Eigen/Dense>
#include <array>
#include <iostream>

namespace engine {

namespace navigation {
    enum Axis {
        X, Y, Z, Dim
    };

    struct MultiIndex {
    private:
        
        void computeStride() {
            stride[0] = 1;
            for(int i = 1; i < Axis::Dim; i++) {
                stride[i] = stride[i-1] * size[i-1];
            }
            nMax = stride[Axis::Dim-1] * size[Axis::Dim-1];
        }
    protected:
        using IndexArray = std::array<int, Axis::Dim>;
        int idx = 0;
        IndexArray size;
        IndexArray stride;
        int nMax = 0;

    public:
        MultiIndex() {}
        MultiIndex(IndexArray _size) : size(_size) {
            computeStride();
        }
        MultiIndex(IndexArray _size, IndexArray _initializer) : size(_size) {
            computeStride();
            for(int i = 0; i < Axis::Dim; i++) {
                idx += _initializer[i] * stride[i];
            }
        }

        MultiIndex& operator++(int) {
            idx++;
            return *this;
        }

        MultiIndex& operator--(int) {
            idx--;
            return *this;
        }

        MultiIndex& operator+=(int i) {
            idx += i;
            return *this;
        }

        MultiIndex& operator-=(int i) {
            idx -= i;
            return *this;
        }

        void advance(Axis axis, int n) {
            idx += stride[axis] * n;
        }

        MultiIndex getSucceeding(Axis axis) {
            MultiIndex result = *this;
            result.advance(axis, 1);
            if(result.isValid()) return result;
            return *this;
        }

        int get() const {
            return idx;
        }

        bool overflow() const {
            return idx >= nMax;
        }

        bool underflow() const {
            return idx < 0;
        }

        bool isValid() const {
            return !underflow() && !overflow();
        }

        IndexArray getIndices() const {
            IndexArray result;
            for(int i = 0; i < Axis::Dim; i++) {
                result[i] = (idx / stride[i]) % size[i];
            }
            return result;
        }

        std::vector<MultiIndex> getPreceding() const {
            std::vector<MultiIndex> result;
            for(int i = 0; i < Axis::Dim; i++) {
                MultiIndex idx = *this;
                idx.idx -= stride[i];
                if(idx.isValid()) result.emplace_back(idx);
            }
            return result;
        }

        std::vector<MultiIndex> getSucceeding() const {
            std::vector<MultiIndex> result;
            for(int i = 0; i < Axis::Dim; i++) {
                MultiIndex idx = *this;
                idx.idx += stride[i];
                if(idx.isValid()) result.emplace_back(idx);
            }
            return result;
        }
    };

    struct GridCoord {
        std::array<float, Axis::Dim> coord;

        GridCoord() {}
        GridCoord(Eigen::VectorXf v) {
            for(int i = 0; i < Axis::Dim; i++) {
                coord.at(i) = v(i);
            }    
        }

        GridCoord(MultiIndex idx) {
            auto indices = idx.getIndices();
            for(int i = 0; i < Axis::Dim; i++) {
                coord.at(i) = float(indices.at(i)) - 0.5f;
            }  
        }

        GridCoord(MultiIndex idx, Axis axis) {
            auto indices = idx.getIndices();
            for(int i = 0; i < Axis::Dim; i++) {
                coord.at(i) = float(indices.at(i)) - (axis == i) ? 0 : 0.5f;
            }  
        }

        operator Eigen::VectorXf()  {
            Eigen::VectorXf v(Axis::Dim);
            for(int i = 0; i < Axis::Dim; i++) {
                v(i) = coord.at(i);
            }
            return v;
        }

        GridCoord shift() {
            GridCoord result(*this);
            for(int i = 0; i < Axis::Dim; i++) {
                result.coord[i] -= 0.5f;
            }
            return result;
        }

        GridCoord shift(Axis axis) {
            GridCoord result(*this);
            for(int i = 0; i < Axis::Dim; i++) {
                if(i == axis) continue;
                result.coord[i] -= 0.5f;
            }
            return result;
        }

        float dist(const GridCoord& other) const {
            float sum = 0.f;
            for(int i = 0; i < Axis::Dim; i++) {
                float dist = coord[i] - other.coord[i];
                sum += dist * dist;
            }
            return std::sqrtf(sum);
        }

        GridCoord& operator=(const Eigen::Vector3f& v) {
            for(int i = 0; i < Axis::Dim; i++) {
                coord.at(i) = v(i);
            } 
            return *this;
        }
    };

}

}