use rayon::{ThreadPool, prelude::*};

use crate::{logging::log_value, simulation::{PhysicalConstants, data::SimulationData, pipeline::operator::pressure_solution::{PressureSolver, linear_operator::{LaplaceOperator, LinearOperator}}}};



fn dot(v: &[f32], w: &[f32]) -> f32 {
    v.iter().zip(w.iter()).map(|(x, y)| x * y).sum()
}

fn dot_par(v: &[f32], w: &[f32]) -> f32 {
    v.par_iter()
        .zip(w.par_iter())
        .map(|(x, y)| x * y)
        .sum()
}

#[derive(Debug)]
pub struct CGSolver {
    pub maxit: usize,
    pub tolerance: f32,
    pub linear_operator: Box<dyn LinearOperator>,

    pub pool: ThreadPool,
}

impl CGSolver {
    pub fn new() -> Self {
        Self {
            maxit: 1000,
            tolerance: 0.001f32,
            linear_operator: Box::new(LaplaceOperator {}),
            pool: rayon::ThreadPoolBuilder::new()
                .num_threads(4)
                .build()
                .unwrap(),
        }
    }
}

impl PressureSolver for CGSolver {
    fn compute(
        &self,
        data: &mut SimulationData,
        physical_constants: &PhysicalConstants,
        dt: f32,
    ) -> Result<usize, String> {
        let mut sum = 0.0f32;
        let mut b = data.divergence.data.clone();
        for x in &mut b {
            *x *= physical_constants.density / dt;
            sum += *x;
        }
        let mean: f32 = sum / b.len() as f32;
        for x in &mut b {
            *x -= mean;
            if x.is_nan() {
                return Err("Received Nan values!".to_string());
            }
        }
        for x in &mut data.pressure.data {
            *x = 0.0f32;
        }

        let x = &mut data.pressure.data;
        assert_eq!(x.len(), b.len());

        let n = x.len();

        let mut r = b.to_vec();
        let mut p = r.clone();

        let mut ap = vec![0.0; n];

        let scale = dot(&b, &b).sqrt();

        if scale < 1e-20 {
            println!("Nothing to do");
            let _ = log_value("logs/solver/iteration_count.log", 0.0);
            return Ok(0);
        }

        let mut rs_old = dot(&r, &r);

        self.pool.install(|| {
            for iter in 0..self.maxit {
                puffin::profile_scope!("cg_iteration");
                {
                    puffin::profile_scope!("linear_operator_apply");
                    self.linear_operator.apply_par(&p, &mut ap, &data.grid, &data.is_solid, &data.flow_edge_allowed, &data.flow_edge_count);
                }

                let denom =  {
                    puffin::profile_scope!("dot_product_pAp");
                    dot_par(&p, &ap)
                };

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

                {
                    puffin::profile_scope!("update");
                    x.par_iter_mut()
                        .zip(r.par_iter_mut())
                        .zip(p.par_iter())
                        .zip(ap.par_iter())
                        .for_each(|(((x_i, r_i), &p_i), &ap_i)| {
                            *x_i += alpha * p_i;
                            *r_i -= alpha * ap_i;
                        });
                }
                

                let rs_new = dot_par(&r, &r);

                let error = rs_new.sqrt() / scale;

                let _ = log_value("logs/solver/solver_error.log", error);
                if error < self.tolerance {
                    let _ = log_value("logs/solver/iteration_count.log", iter as f32);
                    return Ok(iter);
                }

                let beta = rs_new / rs_old;
                rs_old = rs_new;

                p.par_iter_mut()
                    .zip(r.par_iter())
                    .for_each(|(p_i, &r_i)| {
                        *p_i = r_i + beta * *p_i;
                    });
            }
            let _ = log_value("logs/solver/iteration_count.log", self.maxit as f32);
            Err("Used maximum iteration count for pressure solver!".to_string())
        })
    }

    fn set_maxit(&mut self, val: usize) {
        self.maxit = val;
    }

    fn set_tolerance(&mut self, val: f32) {
        self.tolerance = val;
    }
}