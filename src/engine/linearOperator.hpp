#pragma once

#include <Eigen/Dense>
#include <Kokkos_Core.hpp>
#include "grid.hpp"

namespace engine
{
    
class LinearOperator {

public:
    virtual void apply(const Eigen::VectorXf& d, Eigen::VectorXf& Ad) const = 0;
    virtual void applyD(const Eigen::VectorXf& d, Eigen::VectorXf& Ad) const = 0;
    virtual void applyDinv(const Eigen::VectorXf& d, Eigen::VectorXf& Ad) const = 0;
    virtual void applyOffDiagonal(const Eigen::VectorXf& d, Eigen::VectorXf& Ad) const = 0;
    virtual void applyU(const Eigen::VectorXf& d, Eigen::VectorXf& Ad) const = 0;
    virtual void applyL(const Eigen::VectorXf& d, Eigen::VectorXf& Ad) const = 0;
};

class LaplacianOperatorCPU : public LinearOperator {
    Grid* grid;
public:
    void setContext(Grid& _grid) { grid = &_grid; }

    void apply(const Eigen::VectorXf& d, Eigen::VectorXf& Ad) const override;
    void applyD(const Eigen::VectorXf& d, Eigen::VectorXf& Ad) const override;
    void applyDinv(const Eigen::VectorXf& d, Eigen::VectorXf& Ad) const override;
    void applyOffDiagonal(const Eigen::VectorXf& d, Eigen::VectorXf& Ad) const override;
    void applyU(const Eigen::VectorXf& d, Eigen::VectorXf& Ad) const override;
    void applyL(const Eigen::VectorXf& d, Eigen::VectorXf& Ad) const override;

    void test();
};

class LaplacianOperatorGPU : public LinearOperator {
    std::vector<std::vector<float>> matrixEntries;

    ~LaplacianOperatorGPU() {
        deinit();
    }
public:
    void setContext(Grid& _grid);

    void init();
    void deinit();

    void apply(const Eigen::VectorXf& d, Eigen::VectorXf& Ad) const override;
    void applyD(const Eigen::VectorXf& d, Eigen::VectorXf& Ad) const override;
    void applyDinv(const Eigen::VectorXf& d, Eigen::VectorXf& Ad) const override;
    void applyOffDiagonal(const Eigen::VectorXf& d, Eigen::VectorXf& Ad) const override;
    void applyU(const Eigen::VectorXf& d, Eigen::VectorXf& Ad) const override;
    void applyL(const Eigen::VectorXf& d, Eigen::VectorXf& Ad) const override;

    void test();
};

} // namespace engine
