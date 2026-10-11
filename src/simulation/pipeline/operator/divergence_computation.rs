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

        for cell_idx in context.total_cell_iter() {
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

                sum += (right - left) / context.get_delta_x(axis);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::data::{SimulationData, grid::{Grid2D, multi_index::{CellIndex, FaceIndex, FromIndex}}};

    fn uniform_empty_grid() -> (SimulationData, Grid2D) {
        let topology = Grid2D::new([3, 3], [0.05, 0.1]);
        let data = SimulationData::new(&topology);
        (data, topology)
    }

    #[test]
    fn test_divergence_computation_null() {
        let (mut empty_grid, mut topology) = uniform_empty_grid();
        let mut test_operator = DivergenceComputation{};

        test_operator.execute(&mut empty_grid, &mut topology).unwrap();

        for d in empty_grid.divergence.data {
            assert_eq!(d, 0.0);
        }
    }

    #[test]
    fn test_divergence_computation() {
        let (mut test_grid, mut topology) = uniform_empty_grid();
        test_grid.velocity[0].set(&topology, &FaceIndex::from_index([1, 1]), 1.0);
        test_grid.velocity[0].set(&topology, &FaceIndex::from_index([2, 1]), 1.0);
        let mut test_operator = DivergenceComputation{};

        test_operator.execute(&mut test_grid, &mut topology).unwrap();

        assert_eq!(*test_grid.divergence.get(&topology, &CellIndex::from_index([0, 1])), -1.0 / topology.get_delta_x(0));
        assert_eq!(*test_grid.divergence.get(&topology, &CellIndex::from_index([1, 1])), 0.0);
        assert_eq!(*test_grid.divergence.get(&topology, &CellIndex::from_index([2, 1])), 1.0 / topology.get_delta_x(0));
    }
}