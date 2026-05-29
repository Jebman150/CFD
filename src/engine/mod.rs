pub mod grid;
pub mod state;
pub mod solver;

use solver::Solver;

use grid::Grid2D;
use state::State2D;

use crate::engine::{grid::multi_index::{CellIndex, FaceIndex}, solver::linear_operator::PoissonOperator, state::field::{CellType, Field}};

pub struct Engine {
    solver: Solver<solver::gpu_solver::cg_solver::CGSolver>,
    pub grid: Grid2D,

    dt: f32,
    density: f32,
}

impl Engine {
    pub fn new(config_path: &str) -> Self {
        let grid_topology = Grid2D::new(config_path.to_owned() + "grid_topology.toml");

        Self {
            solver: Solver::new(config_path.to_owned() + "solving_algorithm.toml"),
            grid: grid_topology,
            dt: 0.01,
            density: 1.0,
        }
    }

    pub fn initialize_grid(&mut self, state: &mut State2D) {
        self.set_solid_border(state);

        self.set_solid_obstacle(state);

        let laplacian = PoissonOperator::new(&self.grid, &state.cell_types);
        self.solver.set_operator(laplacian);
    }


    pub fn apply_boundary_condition(&self, state: &mut State2D) {
        self.apply_domain_boundary_condition(state);
        self.apply_solid_cells_condition(state);
    }

    pub fn spawn_smoke(&self, state: &mut State2D) {
        for cell_idx in self.grid.cell_iter() {
            if cell_idx.idx[0] != 1 || 
                cell_idx.idx[1] < (self.grid.get_outer_size()[1] as f32 / 4.0) as usize || 
                cell_idx.idx[1] >= (self.grid.get_outer_size()[1] as f32 * 3.0 / 4.0) as usize 
            {
                continue;
            }

            state.smoke.set(&self.grid, &cell_idx, 1.0);
        }
    }

    pub fn advect(&self, velocity_field: &[Field<f32, FaceIndex<2>>; 2], field: &mut Field<f32, CellIndex<2>>) {
        let mut new_field = field.clone();
        for cell_idx in self.grid.cell_iter() {
            let pos = self.grid.cell_world_position(&cell_idx);

            let vel = self.sample_velocity(&velocity_field, pos);

            let prev_pos = [
                pos[0] - self.dt * vel[0],
                pos[1] - self.dt * vel[1]
            ];

            let advected = field.sample(&self.grid, prev_pos);

            new_field.set(&self.grid, &cell_idx, advected);
        }
        *field = new_field;
    }

    pub fn advect_velocities(&self, state: &mut State2D) {
        let mut new_velocity = [
            state.velocity[0].clone(),
            state.velocity[1].clone(),
        ];

        for axis in 0..2 {
            for mut face_idx in self.grid.face_iter(axis) {
                face_idx.axis = axis;

                let pos = self.grid.face_world_position(&face_idx);

                let vel = self.sample_velocity(
                        &state.velocity,
                        pos,
                    );

                let prev_pos = [
                    pos[0] - self.dt * vel[0],
                    pos[1] - self.dt * vel[1],
                ];

                let advected = state.velocity[axis].sample(&self.grid, axis, prev_pos);

                new_velocity[axis]
                    .set(&self.grid, &face_idx, advected);
            }
        }

        state.velocity = new_velocity;
    }

    fn sample_velocity(
        &self,
        velocity_field: &[Field<f32, FaceIndex<2>>; 2],
        pos: [f32; 2],
    ) -> [f32; 2] {

        [
            velocity_field[0].sample(&self.grid, 0, pos),
            velocity_field[1].sample(&self.grid, 1, pos)
        ]
    }

    pub fn compute_pressure(&self, state: &mut State2D) {
        let divergence = self.compute_divergence(state);

        let rhs: Vec<f32> = divergence.iter().map(|x| x * (self.density / self.dt)).collect();

        let mean: f32 = rhs.iter().sum::<f32>() / rhs.len() as f32;
        let cleaned_rhs: Vec<f32> = rhs.into_iter().map(|x| x - mean).collect();

        let mut pressure = vec![0.0; divergence.len()]; 
        self.solver.solve(&mut pressure, &cleaned_rhs);

        for cell_idx in self.grid.cell_iter() {
            state.pressure.set(&self.grid, &cell_idx, pressure[self.grid.linearize_inner_cell_index(&cell_idx)]);
        }

        self.update_velocities(state);
    }

    fn compute_divergence(&self, state: &State2D) -> Vec<f32> {
        let mut result = Vec::new();

        for cell_idx in self.grid.cell_iter() {
            let mut sum = 0.0;

            for axis in 0..2 {
                let adjacent_faces = self.grid.cell_faces(&cell_idx, axis);
                sum += (
                    state.velocity[axis].get(&self.grid, &adjacent_faces[1]) - 
                    state.velocity[axis].get(&self.grid,&adjacent_faces[0])
                ) / self.grid.get_delta_x();
            }

            result.push(sum);
        }
        result
    }

    fn update_velocities(&self, state: &mut State2D) {
        for axis in 0..2 {
            for mut face_idx in self.grid.face_iter(axis) {
                face_idx.axis = axis;
                let old_vel = state.velocity[axis].get(&self.grid, &face_idx);
                let new_vel = old_vel + (self.dt / self.density) * self.get_pressure_gradient(state, &face_idx);

                state.velocity[axis].set(&self.grid, &face_idx, new_vel);
            }
        }
    }

    fn get_pressure_gradient(&self, state: &State2D, face_idx: &FaceIndex<2>) -> f32 {
        let adjacent_cells = self.grid.face_adjacent_cells(&face_idx);
        (state.pressure.get(&self.grid, &adjacent_cells[1]) - state.pressure.get(&self.grid, &adjacent_cells[0])) / self.grid.get_delta_x()
    }

    fn apply_domain_boundary_condition(&self, state: &mut State2D) {
        for axis in 0..2 {
            for start in [true, false] {
                for mut face_idx in self.grid.border_face_iter(axis, start) {
                    face_idx.axis = axis;
                    let cond_value = self.grid.get_condition(axis, start);

                    state.velocity[axis].set(&self.grid, &face_idx, cond_value);
                }
            }
        }
    }

    fn apply_solid_cells_condition(&self, state: &mut State2D) {
        for cell_idx in self.grid.total_cell_iter() {
            if *state.cell_types.get(&self.grid, &cell_idx) == CellType::Solid {
                self.set_cell_faces(state, &cell_idx, 0.0);
            }
        }
    }

    fn set_cell_faces(&self, state: &mut State2D, cell_idx: &CellIndex<2>, value: f32) {
        for axis in 0..2 {
            for face_idx in self.grid.get_face_indices(cell_idx, axis) {
                state.velocity[axis].set(&self.grid, &face_idx, value);
            }
        }
    }

    fn set_solid_border(&self, state: &mut State2D) {
        for border_iter in self.grid.solid_border_iter() {
            for cell_idx in border_iter {
                state.cell_types.set(&self.grid, &cell_idx,  CellType::Solid);
            }
        }
    }

    fn set_solid_obstacle(&self, state: &mut State2D) {
        for cell_idx in self.grid.cell_iter() {
            if cell_idx.idx[0] < (self.grid.get_outer_size()[0] as f32 * 2.0 / 16.0) as usize ||
                cell_idx.idx[0] >= (self.grid.get_outer_size()[0] as f32 * 4.0 / 16.0) as usize ||
                cell_idx.idx[1] < (self.grid.get_outer_size()[1] as f32 * 7.0 / 16.0) as usize ||
                cell_idx.idx[1] >= (self.grid.get_outer_size()[1] as f32 * 9.0 / 16.0) as usize
            {
                continue;
            }

            state.cell_types.set(&self.grid, &cell_idx, CellType::Solid);
        }
    }
}
