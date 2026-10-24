use crate::engine::{grid::{Grid2D, multi_index::CellIndex}, state::field::{CellType, Field}};

pub trait LinearOperator {
    fn apply(
        &self,
        x: &[f32],
        y: &mut [f32],
    );
    /*pub fn apply_diag(&self, x: &[f32]) -> Vec<f32>;
    pub fn apply_diag_inv(&self, x: &[f32]) -> Vec<f32>;
    pub fn apply_upper(&self, x: &[f32]) -> Vec<f32>;
    pub fn apply_lower(&self, x: &[f32]) -> Vec<f32>;*/
}

#[derive(Clone)]
pub struct PoissonOperator {
    pub diag: Vec<f32>,
    pub off_diag: Vec<f32>,
    pub stride: Vec<u32>,

    pub n: usize,
    pub ndim: usize,
}

impl PoissonOperator {
    pub fn empty() -> Self {
        Self {
            n: 0,
            ndim: 0,
            diag: vec![],
            off_diag: vec![],
            stride: vec![]
        }
    }

    pub fn new(grid: &Grid2D, cell_types: &Field<CellType, CellIndex<2>>) -> Self {
        let n = grid.get_cell_count();
        let ndim = 2;

        let mut diag = vec![0.0; n];
        let mut off_diag = vec![0.0; n*ndim];

        let c = 1.0 / (grid.get_delta_x() * grid.get_delta_x());

        for cell_idx in grid.cell_iter() {
            let matrix_index = grid.linearize_inner_cell_index(&cell_idx);

            if *cell_types.get(grid, &cell_idx) == CellType::Solid {
                diag[matrix_index] = 1.0;
                continue;
            }

            for (axis, adjacent_idx) in grid.succeeding_cells(&cell_idx) {
                let matrix_adjacent = grid.linearize_inner_cell_index(&adjacent_idx);
                if matrix_adjacent >= n {
                    continue;
                }

                if *cell_types.get(grid, &adjacent_idx) == CellType::Solid {
                    continue;
                }

                off_diag[matrix_index * ndim + axis] = -c;
                diag[matrix_index] += c;
                diag[matrix_adjacent] += c;
            }
        }

        let mut stride = Vec::new();
        stride.push(1);
        for axis in 1..ndim {
            let last_stride = stride[axis-1];
            let next_stride = last_stride * grid.get_size(axis-1) as u32;
            stride.push(next_stride);
        }

        Self {
            n: n,
            ndim: ndim,
            diag: diag,
            off_diag: off_diag,
            stride: stride
        }
    }

    pub fn print(&self) {
        for n in 0..self.n {
            print!{"{}: " , n};
            for axis in (0..self.ndim).rev() {
                let i = n.checked_sub(self.stride[axis] as usize);
                match i {
                    None => print!(" 0.0 "),
                    Some(j) => print!(" {:} ", self.off_diag[j * self.ndim + axis]),
                };
            }
            print!("{}", self.diag[n]);
            for axis in 0..self.ndim {
                print!(" {:}", self.off_diag[n * self.ndim + axis]);
            }
            println!();
        }
    }
}

impl PoissonOperator {
    pub fn apply(
        &self,
        x: &[f32],
        y: &mut [f32],
    ) {
        y.fill(0.0);

        for i in 0..self.n {
            y[i] += self.diag[i] * x[i];

            for axis in 0..self.ndim {
                let coeff = self.off_diag[i * self.ndim + axis];

                if coeff != 0.0 {
                    let j = i + self.stride[axis] as usize;

                    if j < self.n {
                        y[i] += coeff * x[j];
                        y[j] += coeff * x[i];
                    }
                }
            }
        }
    }

    /*fn apply_diag(&self, x: &[f32]) -> Vec<f32> {
        
    }

    fn apply_diag_inv(&self, x: &[f32]) -> Vec<f32> {
        
    }

    fn apply_lower(&self, x: &[f32]) -> Vec<f32> {
        
    }

    fn apply_upper(&self, x: &[f32]) -> Vec<f32> {
        
    }*/
}

#[cfg(test)]
mod tests {

    use crate::{engine::{Engine, state::State2D}, rand::gen_range}; // 0.6.5

    fn generate_random_vec(n: usize) -> Vec<f32> {
        let vals: Vec<f32> = (0..n as i32).map(|_| gen_range(0, 100) as f32 / 100.0).collect();

        println!("{:?}", vals);
        vals
    }

    fn dot(v: &[f32], w: &[f32]) -> f32 {
        v.iter().zip(w.iter()).map(|(x, y)| x + y).sum()
    }

    use super::*;

    #[test]
    fn test_poisson_operator() {
        let mut engine = Engine::new("config/test/");
        let mut state = State2D::new(&engine.grid);
        engine.initialize_grid(&mut state);

        let n: usize = engine.grid.get_cell_count();
        let test_x = generate_random_vec(n);
        let test_y = generate_random_vec(n);

        let op = PoissonOperator::new(&engine.grid, &state.cell_types);
        op.print();

        let mut op_x: Vec<f32> = vec![0.0; n];
        let mut op_y: Vec<f32> = vec![0.0; n];

        op.apply(&test_x, &mut op_x);
        op.apply(&test_y, &mut op_y);
        
        let a = dot(&op_x, &test_y);
        let b = dot(&op_y, &test_x);

        println!("Diff: {}, {}", a, b);
        assert!((a - b).abs() < 0.1);
    }
}