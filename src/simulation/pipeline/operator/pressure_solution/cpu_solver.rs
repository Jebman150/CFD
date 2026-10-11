use std::time::Instant;

use crate::simulation::pipeline::operator::divergence_computation;
use crate::simulation::{data::SimulationData, pipeline::operator::SimulationOperator};
use crate::logging::{log_solver_value, start_solver_log}; 

use super::PressureSolver;

fn dot(v: &[f32], w: &[f32]) -> f32 {
    v.iter().zip(w.iter()).map(|(x, y)| x * y).sum()
}

fn mul(v: &[f32], w: &[f32]) -> Vec<f32> {
    v.iter().zip(w.iter()).map(|(x, y)| x * y).collect()
}

pub fn apply_poisson(
    nx: usize,
    x: &[f32],
    y: &mut [f32],
) {
    y.fill(0.0);

    let dx = 0.01;
    let diag = 4.0 / (dx * dx);
    let offdiag = -1.0 / (dx * dx);

    for j in 0..nx {
        for i in 0..nx {
            let idx = j * nx + i;

            // Center node contribution (4 * x_i)
            y[idx] += diag * x[idx];

            // Left neighbor (i - 1)
            if i > 0 {
                y[idx] += offdiag * x[idx - 1];
            }

            // Right neighbor (i + 1)
            if i + 1 < nx {
                y[idx] += offdiag * x[idx + 1];
            }

            // Bottom neighbor (j - 1)
            if j > 0 {
                y[idx] += offdiag * x[idx - nx];
            }

            // Top neighbor (j + 1)
            if j + 1 < nx {
                y[idx] += offdiag * x[idx + nx];
            }
        }
    }
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

impl SimulationOperator for CGSolver {
    fn execute(
        &mut self,
        data: &mut SimulationData,
        _context: &mut crate::simulation::data::grid::Grid2D,
    ) -> Result<u128, String> {
        match start_solver_log("log/solver/solver_error.log") {
            Ok(()) => {},
            Err(err) => return Err(format!("Could not start log line: {}", err)),
        };
        match start_solver_log("log/solver/abs_divergence.log") {
            Ok(()) => {},
            Err(err) => return Err(format!("Could not start log line: {}", err)),
        };
        let start = Instant::now();
        let dt = 0.01;
        self.compute(data, dt)?;
        Ok(start.elapsed().as_micros())
    }

    fn get_name(&self) -> &'static str {
        "CG Pressure Solver"
    }
}

fn greatest_distance(vec: &[f32]) -> Option<f32> {
    if vec.is_empty() {
        return None;
    }

    let mut iter = vec.iter().copied().filter(|x| !x.is_nan());

    let first = iter.next()?;
    
    let (min, max) = iter.fold((first, first), |(min, max), x| {
        (min.min(x), max.max(x))
    });

    Some(max - min)
}

impl PressureSolver for CGSolver {
    fn compute(
        &mut self,
        data: &mut SimulationData,
        dt: f32,
    ) -> Result<usize, String> {
        let density = 1.0f32;
        let mut sum = 0.0f32;
        let mut b = data.divergence.data.clone();
        //println!("Divergence before preprocessing: {:?}", b);
        for x in &mut b {
            *x *= density / dt;
            sum += *x;
        }
        //println!("Divergence after scaling: {:?}", b);
        let mean: f32 = sum / b.len() as f32;
        for x in &mut b {
            *x -= mean;
            if x.is_nan() {
                return Err("Received Nan values!".to_string());
            }
        }
        //println!("Divergence after shifting: {:?}", b);
        let _ = log_solver_value("log/solver/abs_divergence.log", greatest_distance(&data.divergence.data).unwrap());
        for x in &mut data.pressure.data {
            *x = 0.0f32;
        }

        let x = &mut data.pressure.data;
        assert_eq!(x.len(), b.len());

        let n = x.len();
        let n_xy = x.len().isqrt();

        x.fill(0.0);

        let mut r = b.to_vec();
        let mut p = r.clone();

        let mut ap = vec![0.0; n];

        let scale = dot(&b, &b).sqrt();

        if scale < 1e-20 {
            println!("Nothing to do");
            return Ok(0);
        }

        let mut rs_old = dot(&r, &r);
        for iter in 0..self.maxit {
            apply_poisson(n_xy, &p, &mut ap);

            let denom = dot(&p, &ap);

            if !denom.is_finite() || denom <= 0.0 {
                return Err(format!(
                    "CG breakdown at iteration {}: p^T A p = {}",
                    iter, denom
                ));
            }

            let alpha = rs_old / denom;

            if !alpha.is_finite() {
                return Err(format!(
                    "Invalid alpha at iteration {}: {}",
                    iter, alpha
                ));
            }

            for i in 0..n {
                x[i] += alpha * p[i];
                r[i] -= alpha * ap[i];
            }

            let rs_new = dot(&r, &r);

            let error = rs_new.sqrt() / scale;

            let _ = log_solver_value("log/solver/solver_error.log", error);
            if error < self.tolerance {
                return Ok(iter);
            }

            let beta = rs_new / rs_old;
            rs_old = rs_new;

            for i in 0..n {
                p[i] = r[i] + beta * p[i];
            }
        }
        Err("Used maximum iteration count for pressure solver!".to_string())
    }

    fn set_maxit(&mut self, val: usize) {
        self.maxit = val;
    }

    fn set_tolerance(&mut self, val: f32) {
        self.tolerance = val;
    }
}