#include "pressureSolver.hpp"

#include <Eigen/IterativeLinearSolvers>
#include <chrono>
#include <iostream>

Eigen::VectorXf PressureSolver::computePressure(const Eigen::VectorXf& divergence, const Eigen::SparseMatrix<float>& A, const Eigen::IncompleteCholesky<float>& ichol) {
    auto start = std::chrono::high_resolution_clock::now();

    Eigen::VectorXf p = Eigen::VectorXf::Zero(divergence.size());
    Eigen::VectorXf r = divergence;
    if(r.norm() < tolerance) return p;
    Eigen::VectorXf z = ichol.solve(r);  //auxiliary vector
    Eigen::VectorXf s = z;      //search vector
    float rho = r.dot(z);
    if(std::abs(rho) < 1e-20)   //invalid state
        return p;
    int iteration = 0;
    while(iteration < maxit) {
        auto q = A * s;
        float alpha = rho / (q.dot(s));
        p += alpha * s;
        r -= alpha * q;
        if(r.norm() < tolerance) {
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<float, std::milli> duration = end - start;
            std::cout << "Took " << iteration << " iterations " << std::endl;
            std::cout << "Finished in " << duration.count() << "ms" << std::endl;
            return p;
        }
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