use std::time::Instant;

use crate::simulation::data::PhysicalConstants;
use crate::simulation::cpu_backend::{
    data::{
        CPUSimulationData,
        field::FaceField2F
    },
    pipeline::operator::SimulationOperator
};

pub struct AdvectVelocity;

fn sample_velocity(
        velocity_field: &[FaceField2F; 2],
        pos: [f32; 2],
        dx: [f32; 2],
    ) -> [f32; 2] {

        [
            velocity_field[0].sample(pos, dx),
            velocity_field[1].sample(pos, dx)
        ]
    }

impl SimulationOperator for AdvectVelocity {
    fn execute(
        &mut self,
        data: &mut CPUSimulationData,
        _physical_constants: &PhysicalConstants,
        dt: f32,
    ) -> Result<u128, String> {
        let start = Instant::now();
        puffin::profile_scope!("advect_velocity");
        let mut new_velocity = [
            data.velocity[0].clone(),
            data.velocity[1].clone(),
        ];

        for axis in 0..2 {
            for face_idx in data.grid.all_faces(axis) {
                let pos = data.grid.face_world_position(&face_idx);

                let vel = sample_velocity(
                        &data.velocity,
                        pos,
                        data.grid.get_delta_x()
                    );
                
                let prev_pos = [
                    pos[0] - dt * vel[0],
                    pos[1] - dt * vel[1],
                ];

                let advected = data.velocity[axis].sample(prev_pos, data.grid.get_delta_x());

                new_velocity[axis]
                    .set(&face_idx, advected);
            }
        }

        data.velocity = new_velocity;
        Ok(start.elapsed().as_micros())
    }
    
    fn get_name(&self) -> &'static str {
        "Advect Velocities"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform_empty_grid() -> CPUSimulationData {
        let data = CPUSimulationData::new([20, 10]);
        data
    }

    fn operator() -> AdvectVelocity {
        AdvectVelocity{}
    }

    #[test]
    fn test_advection() {
        let mut test_grid = uniform_empty_grid();
        for face_idx in test_grid.grid.all_faces(0) {
            test_grid.velocity[0].set(&face_idx, 1.0);
        }
        let mut test_operator = operator();

        test_operator.execute(&mut test_grid, &PhysicalConstants::default(), 0.01).unwrap();

        for face_idx in test_grid.grid.border_face_iter(0, true) {
            assert_eq!(test_grid.velocity[0].get(&face_idx), 0.0f32);
        }
        for face_idx in test_grid.grid.all_faces(0) {
            assert_eq!(test_grid.velocity[0].get(&face_idx), 1.0f32);
        }
    }
}