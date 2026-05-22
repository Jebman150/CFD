#include "pressureSolver.hpp"

#include <Eigen/IterativeLinearSolvers>
#include <chrono>
#include <iostream>

namespace engine {

Eigen::VectorXf PressureSolver::computePressure(const Eigen::VectorXf& divergence) {
    Eigen::VectorXf p = Eigen::VectorXf::Zero(divergence.size());
    Eigen::VectorXf r = -divergence;

    if(r.norm()/divergence.norm() < tolerance) return p;

    Eigen::VectorXf z(divergence.size());   //auxiliary vector
    //applyPreconditioner(r, z);
    z = r;
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
        //applyPreconditioner(r, z);
        z = r;
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
    for(Indexer3D idxer(gridSize); !idxer.end(); idxer++) {
        MultiIndex idx = idxer.get();
        if(grid->getCellType(idx) == CellType::Solid) {
            Ad[cellIndex(idx)] = d[cellIndex(idx)];
            continue;
        }
        float diag = 0.f;
        float sum = 0.f;

        grid->forEachNeighbour(idx, [this, &diag, &sum, &d](ConstCellView cell) {
            if(cell.type == CellType::Solid) return;
            sum += cell.offset * d[cellIndex(cell.idx.i, cell.idx.j, cell.idx.k)];
            diag += cell.offset;
        });

        Ad(cellIndex(idx)) = diag * d(cellIndex(idx)) - sum;
    }
}

void PressureSolver::applyPreconditioner(const Eigen::VectorXf& v, Eigen::VectorXf& LLTz) {
    auxiliaryVec.setZero();
    LLTz.setZero();
    float t = 0;
    using namespace engine::navigation;
    for(Indexer3D idxer(gridSize); !idxer.end(); idxer++) {
        MultiIndex idx = idxer.get();
        t = v(cellIndex(idx));
        auto precIndices = idx.getPreceding();
        for(int i = 0; i < Axis::Dim; i++) {
            MultiIndex pIdx = precIndices[i];
            if(pIdx.isPositive()) t -= getA(pIdx, static_cast<Axis>(i)) * preconditioner.at(cellIndex(pIdx)) * auxiliaryVec(cellIndex(pIdx));
        }
        auxiliaryVec(cellIndex(idx)) = t * preconditioner.at(cellIndex(idx));
    }

    for(Indexer3D idxer(gridSize, true); !idxer.end(); idxer--) {
        MultiIndex idx = idxer.get();
        t = auxiliaryVec(cellIndex(idx));
        if(idx.i+1 < gridSize.x()) t -= getA(idx, Axis::X) * preconditioner.at(cellIndex(idx)) * LLTz(cellIndex(idx.i+1, idx.j, idx.k));
        if(idx.j+1 < gridSize.y()) t -= getA(idx, Axis::Y) * preconditioner.at(cellIndex(idx)) * LLTz(cellIndex(idx.i, idx.j+1, idx.k));
        if(idx.k+1 < gridSize.z()) t -= getA(idx, Axis::Z) * preconditioner.at(cellIndex(idx)) * LLTz(cellIndex(idx.i, idx.j, idx.k+1));
        LLTz(cellIndex(idx)) = t * preconditioner.at(cellIndex(idx));
    }
}

void PressureSolver::buildPreconditioner() {
    preconditioner.assign(gridSize.x()*gridSize.y()*gridSize.z(), 1.f);

    for(Indexer3D idxer(gridSize); !idxer.end(); idxer++) {
        MultiIndex idx = idxer.get();
        float e = getDiagA(idx);
        auto pInd = idx.getPreceding();
        for(int i = 0; i < Axis::Dim; i++) {
            MultiIndex pIdx = pInd[i];
            if(pIdx.isPositive()) e -=  (getA(pIdx, static_cast<Axis>(i)) * preconditioner.at(cellIndex(pIdx))) *
                                        (getA(pIdx, static_cast<Axis>(i)) * preconditioner.at(cellIndex(pIdx)));
        }
        float s = 0.f;
        if(pInd[0].isPositive()) s +=   getA(pInd[0], Axis::X) * (getA(pInd[0], Axis::Y) + getA(pInd[0], Axis::Z)) * preconditioner.at(cellIndex(pInd[0])) * preconditioner.at(cellIndex(pInd[0]));
        if(pInd[1].isPositive()) s +=   getA(pInd[1], Axis::Y) * (getA(pInd[1], Axis::X) + getA(pInd[1], Axis::Z)) * preconditioner.at(cellIndex(pInd[1])) * preconditioner.at(cellIndex(pInd[1]));
        if(pInd[2].isPositive()) s +=   getA(pInd[2], Axis::Z) * (getA(pInd[2], Axis::X) + getA(pInd[2], Axis::Y)) * preconditioner.at(cellIndex(pInd[2])) * preconditioner.at(cellIndex(pInd[2]));
        e -= tuningConstant * s;
        preconditioner.at(cellIndex(idx)) = 1.f/std::sqrtf(std::max(e, 1e-6f));
    }
}

float PressureSolver::getA(MultiIndex idx, Axis axis) {
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

float PressureSolver::getDiagA(MultiIndex idx) {
    float sum = 0.f;
    sum += static_cast<int>(idx.i > 0) * c.x();
    sum += static_cast<int>(idx.i < gridSize.x()-1) * c.x();
    sum += static_cast<int>(idx.j > 0) * c.y();
    sum += static_cast<int>(idx.j < gridSize.y()-1) * c.y();
    sum += static_cast<int>(idx.k > 0) * c.z();
    sum += static_cast<int>(idx.k < gridSize.z()-1) * c.z();

    return sum;
}

}