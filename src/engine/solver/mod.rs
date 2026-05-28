mod algorithms;
pub mod linear_operator;

use std::fs;
use serde::Deserialize;
use anyhow::Result;

#[derive(Deserialize)]
pub enum Algorithm {
    CGSolver,
}

#[derive(Deserialize)]
struct Config {
    algorithm: Algorithm,
    tolerance: f32,
    maxit: f32,
}

#[derive(Deserialize)]
pub struct Solver {
    algorithm: Algorithm,
    tolerance: f32,
    maxit: usize,
}

impl Solver {
    pub fn new(config: String) -> Result<Self> {
        let contents = fs::read_to_string(config)?;
        let config: Self = toml::from_str(&contents)?;
        Ok(config)
    }

    pub fn solve<A: linear_operator::LinearOperator> (
        &self,
        op: &A,
        x: &mut [f32],
        d: &[f32]
    ) {
        match self.algorithm  {
            Algorithm::CGSolver => algorithms::cg_solver(
                op,
                x,
                d,
                self.maxit,
                self.tolerance
                )
        };
    }
}