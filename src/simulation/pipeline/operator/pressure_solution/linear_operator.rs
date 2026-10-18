use crate::simulation::data::{field::CellField2, grid::{Grid2D, layout::GridLayout}};

use rayon::prelude::*;


pub trait LinearOperator: std::fmt::Debug + Send + Sync {
    fn apply(&self, x: &[f32], out: &mut [f32], grid: &Grid2D, is_solid: &CellField2<bool>, edge_flow_allowed: &CellField2<i8>, edge_flow_count: &[CellField2<i8>]);
    fn apply_par(&self, x: &[f32], out: &mut [f32], grid: &Grid2D, is_solid: &CellField2<bool>, edge_flow_allowed: &CellField2<i8>, edge_flow_count: &[CellField2<i8>]) {
        self.apply(x, out, grid, is_solid, edge_flow_allowed, edge_flow_count)
    }
}

#[derive(Debug)]
pub struct LaplaceOperator {

}

impl LinearOperator for LaplaceOperator {
    fn apply_par(&self, x: &[f32], out: &mut [f32], grid: &Grid2D, is_solid: &CellField2<bool>, edge_flow_allowed: &CellField2<i8>, edge_flow_count: &[CellField2<i8>]) {
        out.fill(0.0);

        let dx = grid.get_delta_x()[0];
        let dy = grid.get_delta_x()[1];

        let dx_2_inv = 1.0 / (dx * dx);
        let dy_2_inv = 1.0 / (dy * dy);

        let nx = is_solid.layout.dimensions()[0];
        let ny = is_solid.layout.dimensions()[1];

        out.par_iter_mut()
            .enumerate()
            .for_each(|(idx, out)| {
                if is_solid.data[idx] {
                    *out = x[idx];
                    return;
                }

                let mut result = 0.0f32;

                let flow_allowed = edge_flow_allowed.data[idx];
                let flow_count_x = edge_flow_count[0].data[idx];
                let flow_count_y = edge_flow_count[1].data[idx];

                result -= 1.0 * dx_2_inv * x[idx - 1] * (flow_allowed >> 0 & 1) as f32;     // Left
                result -= 1.0 * dy_2_inv * x[idx - nx] * (flow_allowed >> 1 & 1) as f32;    // Top
                result -= 1.0 * dx_2_inv * x[idx + 1] * (flow_allowed >> 2 & 1) as f32;     // Right
                result -= 1.0 * dy_2_inv * x[idx + nx] * (flow_allowed >> 3 & 1) as f32;    // Bottom

                result += (flow_count_x as f32 * dx_2_inv + flow_count_y as f32 * dy_2_inv) * x[idx];
                *out = result;
            });
    }
    
    fn apply(&self, x: &[f32], out: &mut [f32], grid: &Grid2D, is_solid: &CellField2<bool>, edge_flow_allowed: &CellField2<i8>, edge_flow_count: &[CellField2<i8>]) {
        out.fill(0.0);

        let dx = grid.get_delta_x()[0];
        let dy = grid.get_delta_x()[1];

        let dx_2_inv = 1.0 / (dx * dx);
        let dy_2_inv = 1.0 / (dy * dy);

        let nx = is_solid.layout.dimensions()[0];
        let ny = is_solid.layout.dimensions()[1];

        out.iter_mut()
            .enumerate()
            .for_each(|(idx, out)| {
                if is_solid.data[idx] {
                    *out = x[idx];
                    return;
                }
                let mut result = 0.0f32;

                let flow_allowed = edge_flow_allowed.data[idx];
                let flow_count_x = edge_flow_count[0].data[idx];
                let flow_count_y = edge_flow_count[1].data[idx];

                result -= 1.0 * dx_2_inv * x[idx - 1] * (flow_allowed >> 0 & 1) as f32;     // Left
                result -= 1.0 * dy_2_inv * x[idx - nx] * (flow_allowed >> 1 & 1) as f32;    // Top
                result -= 1.0 * dx_2_inv * x[idx + 1] * (flow_allowed >> 2 & 1) as f32;     // Right
                result -= 1.0 * dy_2_inv * x[idx + nx] * (flow_allowed >> 3 & 1) as f32;    // Bottom

                result += (flow_count_x as f32 * dx_2_inv + flow_count_y as f32 * dy_2_inv) * x[idx];
                *out = result;
            });
    }
}