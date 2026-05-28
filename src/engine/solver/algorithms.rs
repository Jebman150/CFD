use macroquad::logging::warn;

use super::linear_operator::LinearOperator;

fn dot(v: &[f32], w: &[f32]) -> f32 {
    v.iter().zip(w.iter()).map(|(x, y)| x * y).sum()
}

pub fn cg_solver<A: LinearOperator>(
    op: &A,
    x: &mut [f32],
    b: &[f32],
    maxit: usize,
    tol: f32,
) -> usize {
    assert_eq!(x.len(), b.len());

    let n = x.len();

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

    for iter in 0..maxit {

        op.apply(&p, &mut ap);

        let denom = dot(&p, &ap);

        if denom.abs() < 1e-20 {
            warn!("CG breakdown");
            return iter;
        }

        let alpha = rs_old / denom;

        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }

        let rs_new = dot(&r, &r);

        let error = rs_new.sqrt() / scale;

        if error < tol {
            println!("Took {} iterations", iter);
            return iter;
        }

        let beta = rs_new / rs_old;

        for i in 0..n {
            p[i] = r[i] + beta * p[i];
        }

        rs_old = rs_new;
    }

    warn!("Solver exceeded max iteration count");

    maxit
}