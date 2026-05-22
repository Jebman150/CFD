#include "linearOperator.hpp"

namespace engine
{
    
void LaplacianOperatorCPU::apply(const Eigen::VectorXf& d, Eigen::VectorXf& Ad) {
    for(MultiIndex idx(grid.getSize()); !idx.overflow(); idx++) {
        if(grid.getCellType(idx) == CellType::Solid) {
            Ad[idx.get()] = d[idx.get()];
            continue;
        }
        float diag = 0.f;
        float sum = 0.f;

        grid.forEachNeighbour(idx, [this, &diag, &sum, &d](ConstCellView cell) {
            if(cell.type == CellType::Solid) return;
            sum += cell.offset * d[cell.idx.get()];
            diag += cell.offset;
        });

        Ad(idx.get()) = diag * d(idx.get()) - sum;
    }
}

} // namespace engine
