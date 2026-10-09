use std::time::Instant;

use crate::simulation::data::SimulationData;

use super::Solver;



fn dot(v: &[f32], w: &[f32]) -> f32 {
    v.iter().zip(w.iter()).map(|(x, y)| x * y).sum()
}

fn mul(v: &[f32], w: &[f32]) -> Vec<f32> {
    v.iter().zip(w.iter()).map(|(x, y)| x * y).collect()
}

pub struct CGSolver {
    pub maxit: usize,
    pub tolerance: f32,
}

impl CGSolver {
    pub fn new() -> Self {
        Self {
            maxit: 1000,
            tolerance: 0.01f32,
        }
    }
}

impl Solver for CGSolver {
    fn compute(
        &mut self,
        data: &mut SimulationData,
        dt: f32,
    ) -> usize {
        println!("Executed cg solver");
        self.maxit
    }

    fn set_maxit(&mut self, val: usize) {
        self.maxit = val;
    }

    fn set_tolerance(&mut self, val: f32) {
        self.tolerance = val;
    }
}