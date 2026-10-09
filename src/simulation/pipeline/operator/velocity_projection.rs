use std::time::Instant;

use crate::simulation::{data::grid::multi_index::FaceIndex, pipeline::operator::SimulationOperator};


pub struct VelocityProjector;

impl VelocityProjector {
    fn get_pressure_gradient(
        &self,
        data: &crate::simulation::data::SimulationData,
        context: &mut crate::simulation::data::grid::Grid2D,
        face_idx: &FaceIndex<2>
    ) -> f32 {
        let adjacent_cells = context.face_adjacent_cells(&face_idx);
        (data.pressure.get(&context, &adjacent_cells[1]) - data.pressure.get(&context, &adjacent_cells[0])) / context.get_delta_x(face_idx.axis)
    }
}

impl SimulationOperator for VelocityProjector {
    fn execute(
        &mut self,
        data: &mut crate::simulation::data::SimulationData,
        context: &mut crate::simulation::data::grid::Grid2D,
    ) -> Result<u128, String> {
        let start = Instant::now();
        let dt = 0.01f32;
        let density = 1.0f32;
        for axis in 0..2 {
            for mut face_idx in context.face_iter(axis) {
                face_idx.axis = axis;
                let old_vel = data.velocity[axis].get(&context, &face_idx);

                let pressure_gradient = self.get_pressure_gradient(data, context, &face_idx);

                if !pressure_gradient.is_finite() || !old_vel.is_finite() {
                    return Err(format!(
                        "Non-finite value during projection: axis={}, gradient={}, velocity={}",
                        axis, pressure_gradient, old_vel
                    ));
                }

                let new_vel = old_vel - (dt / density) * pressure_gradient;

                data.velocity[axis].set(&context, &face_idx, new_vel);
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
}