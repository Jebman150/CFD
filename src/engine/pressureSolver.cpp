#include "pressureSolver.hpp"

#include <Eigen/IterativeLinearSolvers>
#include <iostream>

Eigen::VectorXd PressureSolver::computePressure(const Eigen::VectorXd& divergence, const Eigen::SparseMatrix<double>& A) {
    Eigen::VectorXd p = Eigen::VectorXd::Zero(divergence.size());
    Eigen::VectorXd r = divergence;
    if(r.norm() < tolerance) return p;
    Eigen::IncompleteCholesky<double> ichol;
    ichol.compute(A);
    if(ichol.info() != Eigen::Success)
    {
        std::cerr << "Factorization failed\n";
        return p;
    }
    Eigen::VectorXd z = ichol.solve(r);  //auxiliary vector
    Eigen::VectorXd s = z;      //search vector
    double rho = r.dot(z);
    if(std::abs(rho) < 1e-20)   //invalid state
        return p;
    int iteration = 0;
    while(iteration < maxit) {
        auto q = A * s;
        double alpha = rho / (q.dot(s));
        p += alpha * s;
        r -= alpha * q;
        if(r.norm() < tolerance) return p;
        z = ichol.solve(r);
        auto rhoNew = z.dot(r);
        auto beta = rhoNew / rho;
        s = z + beta*s;
        rho = rhoNew;
        iteration++;
        //std::cout << "It " << iteration << ": " << r.norm() << std::endl;
    }
    std::cout << "WARNING: Maxit reached in pressure solver" << std::endl;
    std::cout << "Last error: " << r.norm() << std::endl;
    return p;
}