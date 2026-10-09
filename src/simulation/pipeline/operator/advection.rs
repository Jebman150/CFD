use std::time::Instant;

use crate::simulation::{data::{SimulationData, field::Field, grid::{Grid2D, multi_index::FaceIndex}}, pipeline::operator::SimulationOperator};

pub struct AdvectVelocity;

fn sample_velocity(
        velocity_field: &[Field<f32, FaceIndex<2>>; 2],
        context: &Grid2D,
        pos: [f32; 2],
    ) -> [f32; 2] {

        [
            velocity_field[0].sample(context, 0, pos),
            velocity_field[1].sample(context, 1, pos)
        ]
    }

impl SimulationOperator for AdvectVelocity {
    fn execute(
        &mut self,
        data: &mut SimulationData,
        context: &mut Grid2D,
    ) -> Result<u128, String> {
        let start = Instant::now();
        let mut new_velocity = [
            data.velocity[0].clone(),
            data.velocity[1].clone(),
        ];

        for axis in 0..2 {
            for mut face_idx in context.face_iter(axis) {
                face_idx.axis = axis;

                let pos = context.face_world_position(&face_idx);

                let vel = sample_velocity(
                        &data.velocity,
                        context,
                        pos
                    );
                
                let dt = 0.01;               // TODO: pass dt
                let prev_pos = [
                    pos[0] - dt * vel[0],
                    pos[1] - dt * vel[1],
                ];

                let advected = data.velocity[axis].sample(context, axis, prev_pos);

                new_velocity[axis]
                    .set(context, &face_idx, advected);
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

    fn uniform_empty_grid() -> (SimulationData, Grid2D) {
        let topology = Grid2D::new([20, 10], [0.05, 0.1]);
        let data = SimulationData::new(&topology);
        (data, topology)
    }

    fn operator() -> AdvectVelocity {
        AdvectVelocity{}
    }

    #[test]
    fn test_advection() {
        let (mut test_grid, mut context) = uniform_empty_grid();
        for mut face_idx in context.face_iter(0) {
            face_idx.axis = 1;
            test_grid.velocity[0].set(&context, &face_idx, 1.0);
        }
        let mut test_operator = operator();

        test_operator.execute(&mut test_grid, &mut context).unwrap();
        test_grid.print_velocity(&context, 0);

        for mut face_idx in context.border_face_iter(0, true) {
            face_idx.axis = 0;
            assert_eq!(*test_grid.velocity[0].get(&context, &face_idx), 0.0f32);
        }
        for mut face_idx in context.face_iter(0) {
            face_idx.axis = 0;
            assert_eq!(*test_grid.velocity[0].get(&context, &face_idx), 1.0f32);
        }
    }
}