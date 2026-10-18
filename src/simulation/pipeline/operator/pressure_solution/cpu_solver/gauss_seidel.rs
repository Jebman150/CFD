use crate::{logging::log_value, simulation::{PhysicalConstants, data::{SimulationData, grid::layout::GridLayout}, pipeline::operator::pressure_solution::PressureSolver}};



#[derive(Debug)]
pub struct GSSolver {
    tolerance: f32,
    maxit: usize,
}

impl GSSolver {
    pub fn new() -> Self {
        Self {
            tolerance: 0.01,
            maxit: 15,
        }
    }
}

impl GSSolver {
    fn max_pressure_residual(
        &self,
        data: &SimulationData,
        b: &[f32],
    ) -> f32 {
        let dx = data.grid.get_delta_x();
        let inv_h2 = [
            1.0 / (dx[0] * dx[0]),
            1.0 / (dx[1] * dx[1]),
        ];

        let mut max_residual = 0.0_f32;

        for cell in data.grid.all_cells() {
            if *data.is_solid.get(&cell) {
                continue;
            }

            let mut ap = 0.0_f32;

            for (axis, n) in data.grid.succeeding_cells(&cell)
                .iter()
                .chain(data.grid.preceeding_cells(&cell).iter())
            {
                if !*data.is_solid.get(&n) {
                    let w = inv_h2[*axis];
                    let p_n = data.pressure.get(&n);
                    let p_i = data.pressure.get(&cell);

                    ap += w * (p_i - p_n);
                }
            }

            let residual = b[data.divergence.layout.linearize(&cell.idx)] - ap;
            max_residual = max_residual.max(residual.abs());
        }

        max_residual
    }
}

impl PressureSolver for GSSolver {
    fn compute(
        &self,
        data: &mut SimulationData,
        physical_constants: &PhysicalConstants,
        dt: f32,
    ) -> Result<usize, String> {
        let mut sum = 0.0f32;
        let mut b = data.divergence.clone();
        for x in &mut b.data {
            *x *= physical_constants.density / dt;
            sum += *x;
        }
        let mean: f32 = sum / b.data.len() as f32;
        for x in &mut b.data {
            *x -= mean;
            if x.is_nan() {
                return Err("Received Nan values!".to_string());
            }
        }
        for x in &mut data.pressure.data {
            *x = 0.0f32;
        }

        assert_eq!(data.pressure.data.len(), b.data.len());

        let dx_2_inv = [
            1.0 / (data.grid.get_delta_x()[0] * data.grid.get_delta_x()[0]),
            1.0 / (data.grid.get_delta_x()[1] * data.grid.get_delta_x()[1]),
        ];

        let mut precision_mode: bool = false;

        for iter in 0..self.maxit {
            puffin::profile_scope!("gs_iteration");
            let mut error = 0.0f32;

            for color in 0..2 {
                for cell_idx in data.grid.all_cells() {
                    if (cell_idx.idx[0] + cell_idx.idx[1]) % 2 != color {
                        continue;
                    }

                    if *data.is_solid.get(&cell_idx) {
                        continue;
                    }

                    let mut sum = 0.0;
                    let mut neighbour_count = [0; 2];
                    for (axis, n) in data.grid
                        .succeeding_cells(&cell_idx)
                        .iter()
                        .chain(data.grid.preceeding_cells(&cell_idx).iter())
                    {
                        if !*data.is_solid.get(&n) {
                            sum += data.pressure.get(&n) * dx_2_inv[*axis];
                            neighbour_count[*axis] += 1;
                        }
                    }
                    
                    let mut denom = 0.0;
                    for axis in 0..2 {
                        denom += neighbour_count[axis] as f32 * dx_2_inv[axis];
                    }

                    if denom == 0.0 {
                        continue;
                    }

                    let new_pressure =
                        (b.get(&cell_idx) + sum)
                        / denom;

                    let old_pressure = data.pressure.get(&cell_idx);

                    error = error.max((new_pressure - old_pressure).abs());

                    data.pressure.set(&cell_idx, new_pressure);
                }
            }

            if !precision_mode {
                let _ = log_value("logs/solver/solver_error.log", error);
            }

            if error < self.tolerance {
                let precise_error = {
                    puffin::profile_scope!("precise_error_computation");
                    self.max_pressure_residual(data, &b.data)
                };
                precision_mode = true;
                if precision_mode {
                    let _ = log_value("logs/solver/solver_error.log", precise_error);
                }
                if precise_error < self.tolerance {
                    let _ = log_value("logs/solver/iteration_count.log", iter as f32);
                    return Ok(iter);
                }
            }
        }
        let _ = log_value("logs/solver/iteration_count.log", self.maxit as f32);
        //Err("Used maximum iteration count for pressure solver!".to_string())
        Ok(self.maxit)
    }

    fn set_maxit(&mut self, val: usize) {
        self.maxit = val;
    }

    fn set_tolerance(&mut self, val: f32) {
        self.tolerance = val;
    }
}