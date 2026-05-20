#pragma once

#include <Eigen/Dense>
#include <array>
#include <iostream>

namespace engine {

namespace navigation {
    enum Axis {
        X, Y, Z, Dim
    };

    struct Index3D {
        int i, j, k;

        std::array<Index3D, 6> getNeighbours() const {
            return std::array<Index3D, 6>{
                Index3D{i-1, j, k},
                Index3D{i+1, j, k},
                Index3D{i, j-1, k},
                Index3D{i, j+1, k},
                Index3D{i, j, k-1},
                Index3D{i, j, k+1},
            };
        }

        std::array<Index3D, 3> getPreceding() const {
            return std::array<Index3D, 3>{
                Index3D{i-1, j, k},
                Index3D{i, j-1, k},
                Index3D{i, j, k-1},
            };
        }

        std::array<Index3D, 6> getCellFaceIndices() const {
            return std::array<Index3D, 6>{
                Index3D{i, j, k},
                Index3D{i+1, j, k},
                Index3D{i, j, k},
                Index3D{i, j+1, k},
                Index3D{i, j, k},
                Index3D{i, j, k+1},
            };
        }

        std::array<Index3D, 2> getFaceCellIndices(Axis axis) const {
            switch (axis) {
                case Axis::X:
                    return std::array<Index3D, 2> {
                        Index3D{i-1, j, k},
                        Index3D{i, j, k}
                    };
                case Axis::Y:
                    return std::array<Index3D, 2> {
                        Index3D{i, j-1, k},
                        Index3D{i, j, k}
                    };
                case Axis::Z:
                    return std::array<Index3D, 2> {
                        Index3D{i, j, k-1},
                        Index3D{i, j, k}
                    };
            }
            return std::array<Index3D, 2> {
                Index3D{i-1, j, k},
                Index3D{i, j, k}
            };
        }

        bool isPositive() const {
            return (i>=0) && (j>=0) && (k>=0);
        }
    };

    class Indexer3D {
        int nx, ny, nz;
        int i = 0, j = 0, k = 0;
        int idx = 0;

    public:
        Indexer3D(int _nx, int _ny, int _nz, bool backward = false) : nx(_nx), ny(_ny), nz(_nz) {
            if(backward) {
                i = nx-1;
                j = ny-1;
                k = nz-1;
            }
        }
        Indexer3D(Eigen::Vector3i size, bool backward = false) : nx(size.x()), ny(size.y()), nz(size.z()) {
            if(backward) {
                i = nx-1;
                j = ny-1;
                k = nz-1;
            }
        }

        Index3D get() const { return {i, j, k}; }
        Indexer3D& operator++() {
            idx++;
            i++;
            if(i >= nx) {
                i = 0;
                j++;
            }
            if(j >= ny) {
                j = 0;
                k++;
            }
            return *this;
        }

        Indexer3D operator++(int) {
            Indexer3D old = *this;
            operator++();
            return old;
        }

        Indexer3D& operator--() {
            idx++;
            i--;
            if(i < 0) {
                i = nx;
                j--;
            }
            if(j < 0) {
                j = ny;
                k--;
            }
            return *this;
        }

        Indexer3D operator--(int) {
            Indexer3D old = *this;
            operator--();
            return old;
        }

        bool end() const { return idx >= (nx * ny * nz); }
        bool operator== (const int& other) { return (idx == other); };
        bool operator!= (const int& other) { return (idx != other); };
        bool operator< (const int& other) { return (idx < other); };
        bool operator> (const int& other) { return (idx > other); };
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

}

}