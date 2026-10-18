use std::time::Instant;

use crate::simulation::{PhysicalConstants, data::grid::multi_index::FaceIndex, pipeline::operator::SimulationOperator};


pub struct VelocityProjector;

impl VelocityProjector {
    fn get_pressure_gradient(
        &self,
        data: &crate::simulation::data::SimulationData,
        face_idx: &FaceIndex<2>
    ) -> f32 {
        let adjacent_cells = data.grid.face_adjacent_cells(&face_idx);
        (data.pressure.get(&adjacent_cells[1]) - data.pressure.get(&adjacent_cells[0])) / data.grid.get_delta_x()[face_idx.axis]
    }
}

impl SimulationOperator for VelocityProjector {
    fn execute(
        &mut self,
        data: &mut crate::simulation::data::SimulationData,
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

#[cfg(test)]
mod tests {
    use crate::simulation::data::{SimulationData, grid::{layout::GridLayout, multi_index::FromIndex}};

    use super::*;

    fn uniform_empty_grid() -> SimulationData {
        let data = SimulationData::new([3, 3]);
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