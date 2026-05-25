#include "linearOperator.hpp"

namespace engine
{
    
void LaplacianOperatorCPU::apply(const Eigen::VectorXf& d, Eigen::VectorXf& Ad) const {
    for(MultiIndex idx = grid->getCellIndex(); !idx.overflow(); idx++) {
        if(grid->getCellType(idx) == CellType::Solid) {
            Ad[idx.get()] = d[idx.get()];
            continue;
        }
        float diag = 0.f;
        float sum = 0.f;

        grid->forEachNeighbour(idx, [this, &diag, &sum, &d](ConstCellView cell) {
            if(cell.type == CellType::Solid) return;
            sum += cell.offset * d[cell.idx.get()];
            diag += cell.offset;
        });

        Ad(idx.get()) = diag * d(idx.get()) - sum;
    }
}

void LaplacianOperatorCPU::applyD(const Eigen::VectorXf& d, Eigen::VectorXf& Ad) const {
    for(MultiIndex idx = grid->getCellIndex(); !idx.overflow(); idx++) {
        if(grid->getCellType(idx) == CellType::Solid) {
            Ad[idx.get()] = d[idx.get()];
            continue;
        }
        float diag = 0.f;

        grid->forEachNeighbour(idx, [this, &diag, &d](ConstCellView cell) {
            if(cell.type == CellType::Solid) return;
            diag += cell.offset;
        });

        Ad(idx.get()) = diag * d(idx.get());
    }
}

void LaplacianOperatorCPU::applyDinv(const Eigen::VectorXf& d, Eigen::VectorXf& Ad) const {
    for(MultiIndex idx = grid->getCellIndex(); !idx.overflow(); idx++) {
        if(grid->getCellType(idx) == CellType::Solid) {
            Ad[idx.get()] = d[idx.get()];
            continue;
        }
        float diag = 0.f;

        grid->forEachNeighbour(idx, [this, &diag, &d](ConstCellView cell) {
            if(cell.type == CellType::Solid) return;
            diag += cell.offset;
        });

        Ad(idx.get()) = d(idx.get()) / diag;
    }
}

void LaplacianOperatorCPU::applyOffDiagonal(const Eigen::VectorXf& d, Eigen::VectorXf& Ad) const {
    for(MultiIndex idx = grid->getCellIndex(); !idx.overflow(); idx++) {
        if(grid->getCellType(idx) == CellType::Solid) {
            Ad[idx.get()] = d[idx.get()];
            continue;
        }
        float sum = 0.f;

        grid->forEachNeighbour(idx, [this, &sum, &d, &idx](ConstCellView cell) {
            if(cell.type == CellType::Solid) return;
            sum += cell.offset * d[cell.idx.get()];
        });

        Ad(idx.get()) = -sum;
    }
}

void LaplacianOperatorCPU::applyU(const Eigen::VectorXf& d, Eigen::VectorXf& Ad) const {
    for(MultiIndex idx = grid->getCellIndex(); !idx.overflow(); idx++) {
        if(grid->getCellType(idx) == CellType::Solid) {
            Ad[idx.get()] = d[idx.get()];
            continue;
        }
        float sum = 0.f;

        grid->forEachSuccessor(idx, [this, &sum, &d, &idx](ConstCellView cell) {
            if(cell.type == CellType::Solid) return;
            sum += cell.offset * d[cell.idx.get()];
        });

        Ad(idx.get()) = -sum;
    }
}

void LaplacianOperatorCPU::applyL(const Eigen::VectorXf& d, Eigen::VectorXf& Ad) const {
    for(MultiIndex idx = grid->getCellIndex(); !idx.overflow(); idx++) {
        if(grid->getCellType(idx) == CellType::Solid) {
            Ad[idx.get()] = d[idx.get()];
            continue;
        }
        float sum = 0.f;

        grid->forEachPredecessor(idx, [this, &sum, &d, &idx](ConstCellView cell) {
            if(cell.type == CellType::Solid) return;
            sum += cell.offset * d[cell.idx.get()];
        });

        Ad(idx.get()) = -sum;
    }
}

void LaplacianOperatorCPU::test() {
    std::cout << "TESTING LAPLACIAN OPERATOR" << std::endl;
    Eigen::VectorXf testV = Eigen::VectorXf::Ones(grid->getCellCount());
    Eigen::VectorXf testResult = Eigen::VectorXf::Ones(grid->getCellCount());
    apply(testV, testResult);
    if(testResult.sum() == 0) std::cout << "Test 1 passed" << std::endl;
}

/* -------- LAPLACIAN OPERATOR GPU ---------- */

void LaplacianOperatorGPU::setContext(Grid& _grid) {
    for(MultiIndex idx = _grid.getCellIndex(); !idx.overflow(); idx++) {
        matrixEntries = std::vector<std::vector<float>>(_grid.getCellCount(), std::vector<float>(_grid.getSize().size() * 2));
        if(_grid.getCellType(idx) == CellType::Solid) {
            matrixEntries.at(idx.get())[0] = 1.f;
            continue;
        }
        float diag = 0.f;

        for(int i = 0; i < _grid.getSize().size(); i++) {
            MultiIndex succ = idx.getSucceeding(static_cast<Axis>(i));
            if(_grid.getCellType(succ) == CellType::Solid) continue;
            matrixEntries.at(idx.get())[i+1] =  -_grid.getDx(static_cast<Axis>(i));
        }

        for(int i = 0; i < _grid.getSize().size(); i++) {
            MultiIndex prec = idx.getPreceeding(static_cast<Axis>(i));
            if(_grid.getCellType(prec) == CellType::Solid) continue;
            matrixEntries.at(idx.get())[i+_grid.getSize().size()+1] =  -_grid.getDx(static_cast<Axis>(i));
        }

        _grid.forEachNeighbour(idx, [this, &diag](ConstCellView cell) {
            if(cell.type == CellType::Solid) return;
            diag += cell.offset;
        });

        matrixEntries.at(idx.get())[0] = diag;
    }

    Kokkos::initialize();
    {
        // Allocate a 1-dimensional view of integers
        Kokkos::View<int*> v("v", 5);
        // Fill view with sequentially increasing values v=[0,1,2,3,4]
        Kokkos::parallel_for("fill", 5, KOKKOS_LAMBDA(int i) { v(i) = i; });
        // Compute accumulated sum of v's elements r=0+1+2+3+4
        int r;
        Kokkos::parallel_reduce(
        "accumulate", 5,
        KOKKOS_LAMBDA(int i, int& partial_r) { partial_r += v(i); }, r);
        // Check the result
        KOKKOS_ASSERT(r == 10);
    }
    Kokkos::printf("Goodbye World\n");
    Kokkos::finalize();
}

void LaplacianOperatorGPU::init() {
    Kokkos::InitializationSettings settings;
    settings.set_num_threads(8);
    Kokkos::initialize(settings);
}

void LaplacianOperatorGPU::deinit() {
    Kokkos::finalize();
}

void LaplacianOperatorGPU::apply(const Eigen::VectorXf& d, Eigen::VectorXf& Ad) const {
    Kokkos::View<const float**> A("laplacian", matrixEntries.size(), matrixEntries[0].size());
    Kokkos::View<const float*> x("x", matrixEntries.size());
    Kokkos::View<float*> result("Ax", matrixEntries.size());
    Kokkos::View<const int[1]> dimensions("dim");
    Kokkos::View<const int*> stride("stride", (matrixEntries[0].size()-1)/2);

    const size_t N = matrixEntries.size();
    const size_t D = matrixEntries[0].size();

    Kokkos::parallel_for(N, KOKKOS_LAMBDA (const ptrdiff_t idx) {
        float sum = 0.f;
        for(int i = 0; i < dimensions(0); i++) {
            if(idx - stride(i)*i > 0) {
                sum += x(idx - stride(i)*i) * A(idx, i+dimensions(0)+1);
            }
            if(idx + stride(i)*i < N) {
                sum += x(idx + stride(i)*i) * A(idx, i+1);
            }
        }
        result(idx) = x(idx) * A(idx, 0) - sum;
    });

}

void LaplacianOperatorGPU::applyD(const Eigen::VectorXf& d, Eigen::VectorXf& Ad) const {
    
}

void LaplacianOperatorGPU::applyDinv(const Eigen::VectorXf& d, Eigen::VectorXf& Ad) const {
    
}

void LaplacianOperatorGPU::applyOffDiagonal(const Eigen::VectorXf& d, Eigen::VectorXf& Ad) const {
    
}

void LaplacianOperatorGPU::applyU(const Eigen::VectorXf& d, Eigen::VectorXf& Ad) const {
    
}

void LaplacianOperatorGPU::applyL(const Eigen::VectorXf& d, Eigen::VectorXf& Ad) const {
    
}

void LaplacianOperatorGPU::test() {
    std::cout << "TESTING LAPLACIAN OPERATOR" << std::endl;
    Eigen::VectorXf testV = Eigen::VectorXf::Ones(matrixEntries.size());
    Eigen::VectorXf testResult = Eigen::VectorXf::Ones(matrixEntries.size());
    apply(testV, testResult);
    if(testResult.sum() == 0) std::cout << "Test 1 passed" << std::endl;
}

} // namespace engine
