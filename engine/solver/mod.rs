pub mod gpu_solver;
pub mod cpu_solver;
pub mod linear_operator;

use std::{fs, time::Instant};
use serde::Deserialize;
use anyhow::Result;

use crate::engine::solver::linear_operator::PoissonOperator;

#[derive(Deserialize)]
struct Config {
    tolerance: f32,
    maxit: usize,
}

pub trait SolvingAlgorithm {
    fn new() -> Self;
    fn init(&mut self, op: linear_operator::PoissonOperator);
    fn solve(
        &self,
        x: &mut [f32],
        b: &[f32],
        maxit: usize,
        tolerance: f32
    ) -> usize;
}

pub struct Solver<Algo: SolvingAlgorithm> {
    config: Config,
    solver: Algo,
}

impl<Algo: SolvingAlgorithm> Solver<Algo> {
    pub fn new(config: String) -> Self {
        Self {
            config: Self::read_config(config).expect("Could not read solving config file"),
            solver: Algo::new(),
        }
    }

    fn read_config(file: String) -> Result<Config> {
        let contents = fs::read_to_string(file)?;
        let config: Config = toml::from_str(&contents)?;
        Ok(config)
    }

    pub fn set_operator(&mut self, op: PoissonOperator) {
        self.solver.init(op);
    }

    pub fn solve (
        &self,
        x: &mut [f32],
        d: &[f32]
    ) {
        let begin = Instant::now();
        let iteration = self.solver.solve(
            x, 
            d, 
            self.config.maxit, 
            self.config.tolerance
        );

        println!("Converged in {} iterations", iteration);
        println!("Solved in {} micros", begin.elapsed().subsec_micros());
    }
}

#[cfg(test)]
mod tests {
    use crate::engine::solver;

use super::*;

    #[test]
    fn compare_solver() {
        

        let mut cpu_solver = Solver::<solver::cpu_solver::CGSolver> {
            config: Config {
                tolerance: 0.01,
                maxit: 100,
            },
            solver: solver::cpu_solver::CGSolver::new()
        };
        let mut gpu_solver = Solver::<solver::gpu_solver::cg_solver::CGSolver> {
            config: Config {
                tolerance: 0.01,
                maxit: 100,
            },
            solver: solver::gpu_solver::cg_solver::CGSolver::new()
        };

        let laplacian = PoissonOperator {
            diag: vec![2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 2.0, 3.0, 2.0],
            off_diag: vec![1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            stride: vec![1, 3],
            n: 9,
            ndim: 2,
        };
        cpu_solver.set_operator(laplacian.clone());
        gpu_solver.set_operator(laplacian);

        let test_case: Vec<f32> = vec![-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let mut cpu_solution = vec![0.0; 9];
        let mut gpu_solution = vec![0.0; 9];

        cpu_solver.solve(&mut cpu_solution, &test_case);
        gpu_solver.solve(&mut gpu_solution, &test_case);

        println!("CPU solution: {:?}", cpu_solution);
        println!("GPU solution: {:?}", gpu_solution);

        for i in 0..9 {
            assert!((cpu_solution[i] - gpu_solution[i]).abs() < 1e-8);
        }
    }
}