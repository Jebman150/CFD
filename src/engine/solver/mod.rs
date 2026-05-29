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