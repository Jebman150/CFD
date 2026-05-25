#pragma once

#include <Eigen/Dense>
#include "linearOperator.hpp"

namespace engine
{
    
class Solver {
protected:
    int maxit = 100;
    float tolerance = 0.01;
public:
    virtual ~Solver() = default;
    virtual Eigen::VectorXf solve(LinearOperator* A, const Eigen::VectorXf& d) = 0;
};

class CGSolver : public Solver {

public:
    Eigen::VectorXf solve(LinearOperator* A, const Eigen::VectorXf& d) override;
};

class RBGSSolver : public Solver {

public:
    Eigen::VectorXf solve(LinearOperator* A, const Eigen::VectorXf& d) override;
};

class JacobiSolver : public Solver {

public:
    Eigen::VectorXf solve(LinearOperator* A, const Eigen::VectorXf& d) override;
};

} // namespace engine
