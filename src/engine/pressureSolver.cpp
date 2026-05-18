#include "pressureSolver.hpp"

#include <Eigen/IterativeLinearSolvers>
#include <chrono>
#include <iostream>

Eigen::VectorXf PressureSolver::computePressure(const Eigen::VectorXf& divergence) {
    Eigen::VectorXf p = Eigen::VectorXf::Zero(divergence.size());
    Eigen::VectorXf r = -divergence;

    if(r.norm()/divergence.norm() < tolerance) return p;

    Eigen::VectorXf z(divergence.size());   //auxiliary vector
    applyPreconditioner(r, z);  
    Eigen::VectorXf s = z;      //search vector
    float sigma = r.dot(z);
    if(std::abs(sigma) < 1e-20)   //invalid state
        return p;
    int iteration = 0;
    while(iteration < maxit) {
        applyA(s, z);
        float denom = z.dot(s);
        if(std::abs(denom) < 1e-20f)
        {
            std::cout << "CG breakdown\n";
            break;
        }
        float alpha = sigma / denom;
        p += alpha * s;
        r -= alpha * z;
        if(r.norm()/divergence.norm() < tolerance) return p;
        applyPreconditioner(r, z);
        auto sigmaNew = r.dot(z);
        auto beta = sigmaNew / sigma;
        s = z + beta*s;
        sigma = sigmaNew;
        iteration++;
        //std::cout << "It " << iteration << ": " << r.norm()/divergence.norm() << std::endl;
    }
    std::cout << "WARNING: Maxit reached in pressure solver" << std::endl;
    std::cout << "Last error: " << r.norm()/divergence.norm() << std::endl;
    return p;
}

void PressureSolver::applyA(const Eigen::VectorXf& d, Eigen::VectorXf& Ad) {
    for(int z = 0; z < gridSize.z(); z++) {
        for(int y = 0; y < gridSize.y(); y++) {
            for(int x = 0; x < gridSize.x(); x++) {
                float diag = 0.f;
                float sum = 0.f;

                //X
                if (x > 0) {
                    sum += c.x() * d[cellIndex(x-1, y, z)];
                    diag += c.x();
                }
                if (x < gridSize.x() - 1) {
                    sum += c.x()  * d[cellIndex(x+1, y, z)];
                    diag += c.x();
                }

                // Y
                if (y > 0) {
                    sum += c.y() * d[cellIndex(x, y-1, z)];
                    diag += c.y();
                }
                if (y < gridSize.y() - 1) {
                    sum += c.y() * d[cellIndex(x, y+1, z)];
                    diag += c.y();
                }

                // Z
                if (z > 0) {
                    sum += c.z() * d[cellIndex(x, y, z-1)];
                    diag += c.z();
                }
                if (z < gridSize.z() - 1) {
                    sum += c.z() * d[cellIndex(x, y, z+1)];
                    diag += c.z();
                }

                Ad(cellIndex(x, y, z)) = diag * d(cellIndex(x, y, z)) - sum;
            }
        }
    }
}

void PressureSolver::applyPreconditioner(const Eigen::VectorXf& v, Eigen::VectorXf& LLTz) {
    auxiliaryVec.setZero();
    LLTz.setZero();
    float t = 0;
    for(int z = 0; z < gridSize.z(); z++) {
        for(int y = 0; y < gridSize.y(); y++) {
            for(int x = 0; x < gridSize.x(); x++) {
                t = v(cellIndex(x, y, z));
                if((x-1) >= 0) t -= getA({x-1, y, z}, Axis::X) * preconditioner.at(cellIndex(x-1, y, z)) * auxiliaryVec(cellIndex(x-1, y, z));
                if((y-1)>= 0) t -= getA({x, y-1, z}, Axis::Y) * preconditioner.at(cellIndex(x, y-1, z)) * auxiliaryVec(cellIndex(x, y-1, z));
                if((z-1) >= 0) t -= getA({x, y, z-1}, Axis::Z) * preconditioner.at(cellIndex(x, y, z-1)) * auxiliaryVec(cellIndex(x, y, z-1));
                auxiliaryVec(cellIndex(x, y, z)) = t * preconditioner.at(cellIndex(x, y, z));
            }
        }
    }

    for(int z = gridSize.z()-1; z >= 0; --z) {
        for(int y = gridSize.y()-1; y >= 0; --y) {
            for(int x = gridSize.x()-1; x >= 0; --x) {
                t = auxiliaryVec(cellIndex(x, y, z));
                if(x+1 < gridSize.x()) t -= getA({x, y, z}, Axis::X) * preconditioner.at(cellIndex(x, y, z)) * LLTz(cellIndex(x+1, y, z));
                if(y+1 < gridSize.y()) t -= getA({x, y, z}, Axis::Y) * preconditioner.at(cellIndex(x, y, z)) * LLTz(cellIndex(x, y+1, z));
                if(z+1 < gridSize.z()) t -= getA({x, y, z}, Axis::Z) * preconditioner.at(cellIndex(x, y, z)) * LLTz(cellIndex(x, y, z+1));
                LLTz(cellIndex(x, y, z)) = t * preconditioner.at(cellIndex(x, y, z));
            }
        }
    }
}

void PressureSolver::buildPreconditioner() {
    preconditioner.assign(gridSize.x()*gridSize.y()*gridSize.z(), 1.f);

    for(int z = 0; z < gridSize.z(); z++) {
        for(int y = 0; y < gridSize.y(); y++) {
            for(int x = 0; x < gridSize.x(); x++) {
                float e = getDiagA({x, y, z});
                if(x-1 >= 0) e -=   (getA({x-1, y, z}, Axis::X) * preconditioner.at(cellIndex(x-1, y, z))) *
                                    (getA({x-1, y, z}, Axis::X) * preconditioner.at(cellIndex(x-1, y, z)));
                if(y-1 >= 0) e -=   (getA({x, y-1, z}, Axis::Y) * preconditioner.at(cellIndex(x, y-1, z))) *
                                    (getA({x, y-1, z}, Axis::Y) * preconditioner.at(cellIndex(x, y-1, z)));
                if(z-1 >= 0) e -=   (getA({x, y, z-1}, Axis::Z) * preconditioner.at(cellIndex(x, y, z-1))) *
                                    (getA({x, y, z-1}, Axis::Z) * preconditioner.at(cellIndex(x, y, z-1)));
                float s = 0.f;
                if(x-1 >= 0) s +=   getA({x-1, y, z}, Axis::X) * (getA({x-1, y, z}, Axis::Y) + getA({x-1, y, z}, Axis::Z)) * preconditioner.at(cellIndex(x-1, y, z)) * preconditioner.at(cellIndex(x-1, y, z));
                if(y-1 >= 0) s +=   getA({x, y-1, z}, Axis::Y) * (getA({x, y-1, z}, Axis::X) + getA({x, y-1, z}, Axis::Z)) * preconditioner.at(cellIndex(x, y-1, z)) * preconditioner.at(cellIndex(x, y-1, z));
                if(z-1 >= 0) s +=   getA({x, y, z-1}, Axis::Z) * (getA({x, y, z-1}, Axis::X) + getA({x, y, z-1}, Axis::Y)) * preconditioner.at(cellIndex(x, y, z-1)) * preconditioner.at(cellIndex(x, y, z-1));
                e -= tuningConstant * s;
                preconditioner.at(cellIndex(x, y, z)) = 1.f/std::sqrtf(std::max(e, 1e-6f));
            }
        }
    }
}

float PressureSolver::getA(Index3D idx, Axis axis) {
    if(idx.i < 0 || idx.j < 0 || idx.k < 0) return 0;
    switch(axis) {
        case Axis::X:
            return idx.i < (gridSize.x() -1) ? -c.x() : 0;
        case Axis::Y:
            return idx.j < (gridSize.y() -1) ? -c.y() : 0;
        case Axis::Z:
            return idx.k < (gridSize.z() -1) ? -c.z() : 0;
    }
    return 0;
}

float PressureSolver::getDiagA(Index3D idx) {
    float sum = 0.f;
    sum += static_cast<int>(idx.i > 0) * c.x();
    sum += static_cast<int>(idx.i < gridSize.x()-1) * c.x();
    sum += static_cast<int>(idx.j > 0) * c.y();
    sum += static_cast<int>(idx.j < gridSize.y()-1) * c.y();
    sum += static_cast<int>(idx.k > 0) * c.z();
    sum += static_cast<int>(idx.k < gridSize.z()-1) * c.z();

    return sum;
}