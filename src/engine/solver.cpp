#include "solver.hpp"

#include <iostream>

namespace engine
{
    
Eigen::VectorXf CGSolver::solve(LinearOperator* A, const Eigen::VectorXf& d) {
    Eigen::VectorXf p = Eigen::VectorXf::Zero(d.size());
    Eigen::VectorXf r = -d;

    if(r.norm()/d.norm() < tolerance) return p;

    Eigen::VectorXf z(d.size());   //auxiliary vector
    //applyPreconditioner(r, z);
    z = r;
    Eigen::VectorXf s = z;      //search vector
    float sigma = r.dot(z);
    if(std::abs(sigma) < 1e-20)   //invalid state
        return p;
    int iteration = 0;
    while(iteration < maxit) {
        z = A->apply(s);
        float denom = z.dot(s);
        if(std::abs(denom) < 1e-20f)
        {
            std::cout << "CG breakdown\n";
            break;
        }
        float alpha = sigma / denom;
        p += alpha * s;
        r -= alpha * z;
        if(r.norm()/d.norm() < tolerance) return p;
        //applyPreconditioner(r, z);
        z = r;
        auto sigmaNew = r.dot(z);
        auto beta = sigmaNew / sigma;
        s = z + beta*s;
        sigma = sigmaNew;
        iteration++;
        //std::cout << "It " << iteration << ": " << r.norm()/d.norm() << std::endl;
    }
    std::cout << "WARNING: Maxit reached in pressure solver" << std::endl;
    std::cout << "Last error: " << r.norm()/d.norm() << std::endl;
    return p;
}

} // namespace engine
