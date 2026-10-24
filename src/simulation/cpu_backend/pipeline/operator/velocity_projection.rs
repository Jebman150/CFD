use std::time::Instant;

use crate::simulation::data::{
    PhysicalConstants,
    grid::multi_index::{
        FaceIndex,
        CellIndex
    }
};
use crate::simulation::cpu_backend::{
    data::{
        CPUSimulationData,
    },
    pipeline::operator::SimulationOperator
};


pub struct VelocityProjector;

impl VelocityProjector {
    fn get_pressure_gradient(
        &self,
        data: &CPUSimulationData,
        face_idx: &FaceIndex<2>
    ) -> f32 {
        let adjacent_cells = data.grid.face_adjacent_cells(&face_idx);
        (data.pressure.get(&adjacent_cells[1]) - data.pressure.get(&adjacent_cells[0])) / data.grid.get_delta_x()[face_idx.axis]
    }
}

impl SimulationOperator for VelocityProjector {
    fn execute(
        &mut self,
        data: &mut CPUSimulationData,
        physical_constants: &PhysicalConstants,
        dt: f32,
    ) -> Result<u128, String> {
        let start = Instant::now();
        puffin::profile_scope!("project_velocity");
        for axis in 0..2 {
            for face_idx in data.grid.interior_faces(axis) {
                let old_vel = data.velocity[axis].get(&face_idx);

                let pressure_gradient = self.get_pressure_gradient(data, &face_idx);

                if !pressure_gradient.is_finite() || !old_vel.is_finite() {
                    return Err(format!(
                        "Non-finite value during projection: axis={}, gradient={}, velocity={}",
                        axis, pressure_gradient, old_vel
                    ));
                }

                let new_vel = old_vel + (dt / physical_constants.density) * pressure_gradient;

                data.velocity[axis].set(&face_idx, new_vel);
            }
        }
        Ok(start.elapsed().as_micros())
    }

    fn get_name(&self) -> &'static str {
        "Velocity projection"
    }
}

pub struct VelocityAbsValCalculation;

impl VelocityAbsValCalculation {
    fn get_velocity(state: &CPUSimulationData, cell_idx: &CellIndex<2>) -> [f32; 2] {
        let x_faces = state.grid.get_face_indices(&cell_idx, 0);
        let vel_x = (state.velocity[0].get(&x_faces[0]) + state.velocity[0].get(&x_faces[1])) * 0.5;
        let y_faces = state.grid.get_face_indices(&cell_idx, 1);
        let vel_y = (state.velocity[1].get(&y_faces[0]) + state.velocity[1].get(&y_faces[1])) * 0.5;
        [vel_x, vel_y]
    }
}

impl SimulationOperator for VelocityAbsValCalculation {
    fn execute(
        &mut self,
        data: &mut CPUSimulationData,
        _physical_constants: &PhysicalConstants,
        _dt: f32,
    ) -> Result<u128, String> {
        let start = Instant::now();
        puffin::profile_scope!("velocity_abs_val_calculation");
        for cell in data.grid.all_cells() {
            let vel = Self::get_velocity(data, &cell);
            data.abs_velocity.set(&cell, vel[0].abs() + vel[1].abs());
        }
        Ok(start.elapsed().as_micros())
    }

    fn get_name(&self) -> &'static str {
        "Velocity absolute value calculation"
    }
}

#[cfg(test)]
mod tests {
    use crate::simulation::data::grid::{layout::GridLayout, multi_index::FromIndex};

    use super::*;

    fn uniform_empty_grid() -> CPUSimulationData {
        let data = CPUSimulationData::new([3, 3]);
        data
    }

    #[test]
    fn test_null_projection() {
        let mut test_data = uniform_empty_grid();
        test_data.pressure.data = vec![5.0; test_data.pressure.layout.total_count()];
        let mut test_operator = VelocityProjector{};

        test_operator.execute(&mut test_data, &PhysicalConstants::default(), 0.01).unwrap();

        for axis in 0..2 {
            for face_idx in test_data.grid.all_faces(axis) {
                assert_eq!(test_data.velocity[axis].get(&face_idx), 0.0);
            }
        }
    }

    #[test]
    fn test_linear_pressure_field() {
        let mut test_data = uniform_empty_grid();
        for cell_idx in test_data.grid.all_cells() {
            test_data.pressure.set(&cell_idx, cell_idx.idx[0] as f32);
        }
        let mut test_operator = VelocityProjector{};

        test_operator.execute(&mut test_data, &PhysicalConstants::default(), 0.01).unwrap();

        let val = test_data.velocity[0].get(&FaceIndex::from_index([1, 1]));
        assert!(val > 0.0);
        for face_idx in test_data.grid.interior_faces(0) {
            assert_eq!(test_data.velocity[0].get(&face_idx), val);
            assert!(test_data.velocity[0].get(&face_idx) > 0.0);
        }
        for face_idx in test_data.grid.all_faces(1) {
            assert_eq!(test_data.velocity[1].get(&face_idx), 0.0);
        }
    }
}