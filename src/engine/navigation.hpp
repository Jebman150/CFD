#pragma once

#include <Eigen/Dense>
#include <array>
#include <iostream>

namespace engine {

namespace navigation {
    enum Axis {
        X, Y, Dim, Z
    };

    struct IndexContext {
        std::vector<int> size;
        std::vector<int> stride;
        int nMax;

        IndexContext() {}
        IndexContext(std::vector<int> _size) : size(_size) {
            computeStride();
        }
    private:
        void computeStride() {
            stride = std::vector<int>(size.size());
            stride[0] = 1;
            for(int i = 1; i < stride.size(); i++) {
                stride[i] = stride[i-1] * size[i-1];
            }
            nMax = stride[stride.size()-1] * size[stride.size()-1];
        }
    };

    struct MultiIndex { 
    protected:
        using IndexArray = std::vector<int>;
        IndexArray indices;
        const IndexContext* context;
    public:
        MultiIndex() {}
        MultiIndex(const IndexContext* _context) : context(_context) {
            if(!context) {
                std::cerr << "ERROR: Using multiindex without context!" << std::endl;
                return;
            }
            indices = IndexArray(context->size.size(), 0);
        }

        MultiIndex(int _initializer, const IndexContext* _context) : MultiIndex(_context) {
            for(int i = 0; i < context->size.size(); i++) {
                indices[i] = (_initializer / context->stride[i]) % context->size[i];
            }
        }
        MultiIndex(IndexArray _initializer, const IndexContext* _context) : MultiIndex(_context) {
            for(int i = 0; i < context->size.size(); i++) {
                indices[i] = _initializer[i];
            }
        }

        MultiIndex& operator++(int) {
            for(int i = 0; i < context->size.size(); i++) {
                indices[i]++;
                if(indices[i] < context->size[i]) break;
                if(i != context->size.size()-1) indices[i] = 0;
            }
            return *this;
        }

        MultiIndex& operator--(int) {
            for(int i = 0; i < context->size.size(); i++) {
                indices[i]--;
                if(indices[i] >= 0) break;
                if(i != context->size.size()-1) indices[i] = context->size[i]-1;
            }
            return *this;
        }

        MultiIndex& operator+=(int n) {
            n = std::abs(n);
            for(int i = 0; i < context->size.size(); i++) {
                indices[i] += n;
                if(indices[i] < context->size[i]) break;
                int overflow = 0;
                while(indices[i] >= context->size[i]) {
                    indices[i] -= context->size[i];
                    overflow++;
                }
                n = overflow;
            }
            return *this;
        }

        MultiIndex& operator-=(int n) {
            n = std::abs(n);
            for(int i = 0; i < context->size.size(); i++) {
                indices[i] -= n;
                if(indices[i] >= 0) break;
                int underflow = 0;
                while(indices[i] < 0) {
                    indices[i] += context->size[i];
                    underflow++;
                }
                n = underflow;
            }
            return *this;
        }

        void advance(Axis axis, int n) {
            indices[axis] += n;
        }

        MultiIndex getSucceeding(Axis axis) {
            MultiIndex result = *this;
            result.indices[axis]++;
            if(result.isValid()) return result;
            return *this;
        }

        MultiIndex getPreceeding(Axis axis) {
            MultiIndex result = *this;
            result.indices[axis]--;
            if(result.isValid()) return result;
            return *this;
        }

        int get() const {
            int idx = 0;
            for(int i = 0; i < context->size.size(); i++) {
                idx += indices[i] * context->stride[i];
            }
            return idx;
        }

        bool overflow() const {
            for(int i = 0; i < context->size.size(); i++) {
                if(indices[i] >= context->size[i]) return true;
            }
            return false;
        }

        bool underflow() const {
            for(int i = 0; i < context->size.size(); i++) {
                if(indices[i] < 0) return true;
            }
            return false;
        }

        bool isValid() const {
            return !underflow() && !overflow();
        }

        IndexArray getIndices() const {
            return indices;
        }

        std::vector<MultiIndex> getPreceeding() const {
            std::vector<MultiIndex> result;
            for(int i = 0; i < context->size.size(); i++) {
                auto ind = getIndices();
                if(--ind[i] >= 0) result.emplace_back(MultiIndex(ind, context));
            }
            return result;
        }

        std::vector<MultiIndex> getSucceeding() const {
            std::vector<MultiIndex> result;
            for(int i = 0; i < context->size.size(); i++) {
                auto ind = getIndices();
                if(++ind[i] < context->size[i]) result.emplace_back(MultiIndex(ind, context));
            }
            return result;
        }

        bool checkContext(const IndexContext* _other) const {
            if(context->nMax != _other->nMax) return false;
            for(int i = 0; i < context->size.size(); i++) {
                if(context->stride[i] != _other->stride[i]) return false;
                if(context->size[i] != _other->size[i]) return false;
            }
            return true;
        }

        void transformToContext(const IndexContext* _context) {
            auto indices = getIndices();
            *this = MultiIndex(indices, _context);
        }

        MultiIndex transformedToContext(const IndexContext* _context) const {
            auto indices = getIndices();
            return MultiIndex(indices, _context);
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
                coord.at(i) = float(indices.at(i)) + 0.5f;
            }  
        }

        GridCoord(MultiIndex idx, Axis axis) {
            auto indices = idx.getIndices();
            for(int i = 0; i < Axis::Dim; i++) {
                coord.at(i) = float(indices.at(i)) + ((axis == i) ? 0 : 0.5f);
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