use std::time::Instant;

use colored::*;
use crate::simulation::pipeline::operator::SimulationOperator;

pub struct DivergenceComputation;

impl SimulationOperator for DivergenceComputation {
    fn execute(
        &mut self,
        data: &mut crate::simulation::data::SimulationData,
        context: &mut crate::simulation::data::grid::Grid2D,
    ) -> Result<u128, String> {
        let start = Instant::now();

        for cell_idx in context.cell_iter() {
            let mut sum = 0.0;

            for axis in 0..2 {
                let adjacent_faces = context.get_face_indices(&cell_idx, axis);
                let right = data.velocity[axis]
                    .get(&context, &adjacent_faces[1]);
                let left = data.velocity[axis]
                    .get(&context, &adjacent_faces[0]);

                let dx = context.get_delta_x(axis);

                if !right.is_finite()
                    || !left.is_finite()
                    || !dx.is_finite()
                    || dx <= 0.0
                {
                    return Err(format!(
                        "Invalid velocity or grid spacing: \
                        cell {:?}, axis {}, left={}, right={}, dx={}",
                        cell_idx, axis, left, right, dx
                    ));
                }

                sum += (left - right) / context.get_delta_x(axis);
                if !sum.is_finite() {
                    return Err(format!(
                        "Divergence overflow: cell {:?}, axis {}",
                        cell_idx, axis
                    ));
                }
            }

            data.divergence.set(context, &cell_idx, sum);
        }
        println!("{}", format!("Computed global divergence: {}", data.divergence.data.iter().sum::<f32>()).green());
        Ok(start.elapsed().as_micros())
    }

    fn get_name(&self) -> &'static str {
        "Divergence computation"
    }
}