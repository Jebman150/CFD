use std::time::Instant;

use crate::engine::solver::linear_operator::PoissonOperator;

use super::SolvingAlgorithm;



fn dot(v: &[f32], w: &[f32]) -> f32 {
    v.iter().zip(w.iter()).map(|(x, y)| x * y).sum()
}

fn mul(v: &[f32], w: &[f32]) -> Vec<f32> {
    v.iter().zip(w.iter()).map(|(x, y)| x * y).collect()
}

pub struct CGSolver { 
    op: Option<PoissonOperator>
}


impl SolvingAlgorithm for CGSolver {
    fn new() -> Self {
        Self {
            op: None
        }
    }

    fn init(&mut self, op: PoissonOperator) {
        self.op = Some(op);
    }

    fn solve(
        &self,
        x: &mut [f32],
        b: &[f32],
        maxit: usize,
        tol: f32,
    ) -> usize {
        assert_eq!(x.len(), b.len());

        let n = x.len();

        let operator = self.op.as_ref().expect("No operator passed");

        x.fill(0.0);

        let mut r = b.to_vec();
        let mut p = r.clone();

        let mut ap = vec![0.0; n];

        let scale = dot(b, b).sqrt();

        if scale < 1e-20 {
            println!("Nothing to do");
            return 0;
        }

        let mut rs_old = dot(&r, &r);

        let mut operator_time = 0;
        let mut dot_time = 0;
        let mut assign_time = 0;

        for iter in 0..maxit {

            let op_start = Instant::now();
            operator.apply(&p, &mut ap);
            //println!("Ap: {:?}", ap);
            operator_time += op_start.elapsed().subsec_micros();

            let dot_start = Instant::now();
            let denom = dot(&p, &ap);
            //println!("pAp: {:?}", denom);
            dot_time += dot_start.elapsed().subsec_micros();

            if denom.abs() < 1e-20 {
                println!("CG breakdown");
                return iter;
            }

            let alpha = rs_old / denom;

            let assign_start = Instant::now();
            for i in 0..n {
                x[i] += alpha * p[i];
                r[i] -= alpha * ap[i];
            }
            //println!("x updated: {:?}", x);
            //println!("r update: {:?}", r);
            assign_time += assign_start.elapsed().subsec_micros();

            let dot_start = Instant::now();
            let rs_new = dot(&r, &r);
            let test: Vec<f32> = r.iter().map(|r| r*r).collect();
            dot_time += dot_start.elapsed().subsec_micros();
            //println!("mul: {:?}", test);
            //println!("dot2: {:?}", rs_new);

            let error = rs_new.sqrt() / scale;

            if error < tol {
                println!("Total operator time {:.2?} micros", operator_time);
                println!("Total dot time {:.2?} micros", dot_time);
                println!("Total assign time {:.2?} micros", assign_time);
                return iter;
            }

            let beta = rs_new / rs_old;
            rs_old = rs_new;

            let assign_start = Instant::now();
            for i in 0..n {
                p[i] = r[i] + beta * p[i];
            }

            //println!("p update: {:?}", p);

            assign_time += assign_start.elapsed().subsec_micros();

            //break;
        }
        maxit
    }
}