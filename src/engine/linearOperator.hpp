#pragma once

#include <Eigen/Dense>
#include "grid.hpp"

namespace engine
{
    
class LinearOperator {

public:
    virtual void apply(const Eigen::VectorXf& d, Eigen::VectorXf& Ad) = 0;
};

class LaplacianOperatorCPU : public LinearOperator {
    Grid& grid;
public:
    LaplacianOperatorCPU(Grid& _grid) : grid(_grid) {}
    void apply(const Eigen::VectorXf& d, Eigen::VectorXf& Ad) override;
};

} // namespace engine
