use std::time::Instant;

use crate::simulation::data::PhysicalConstants;
use crate::simulation::cpu_backend::{
    data::{
        CPUSimulationData,
    },
    pipeline::operator::SimulationOperator
};
use crate::logging::log_value;

pub struct DivergenceComputation;

impl SimulationOperator for DivergenceComputation {
    fn execute(
        &mut self,
        data: &mut CPUSimulationData,
        _physical_constants: &PhysicalConstants,
        _dt: f32,
    ) -> Result<u128, String> {
        let start = Instant::now();
        puffin::profile_scope!("compute_divergence");

        for cell_idx in data.grid.all_cells() {
            let mut sum = 0.0;

            for axis in 0..2 {
                let adjacent_faces = data.grid.get_face_indices(&cell_idx, axis);
                let right = data.velocity[axis]
                    .get(&adjacent_faces[1]);
                let left = data.velocity[axis]
                    .get(&adjacent_faces[0]);

                let dx = data.grid.get_delta_x()[axis];

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

                sum += (right - left) / data.grid.get_delta_x()[axis];
                if !sum.is_finite() {
                    return Err(format!(
                        "Divergence overflow: cell {:?}, axis {}",
                        cell_idx, axis
                    ));
                }
            }

            data.divergence.set(&cell_idx, sum);
        }
        Ok(start.elapsed().as_micros())
    }

    fn get_name(&self) -> &'static str {
        "Divergence computation"
    }
}

pub struct DivergenceChecker;

impl SimulationOperator for DivergenceChecker {
    fn execute(
        &mut self,
        data: &mut CPUSimulationData,
        _physical_constants: &PhysicalConstants,
        _dt: f32,
    ) -> Result<u128, String> {
        let mut max = 0.0;
        for d in &data.divergence.data {
            if (*d).abs() >= 0.1 {
                return Err(format!(
                    "Divergence check failed: div(v) = {}",
                    d.abs()
                ));
            }
            if d.abs() >= max {
                max = d.abs();
            }
        }
        let _ = log_value("logs/divergence_checker/max_divergence.log", max);
        Ok(0)
    }

    fn get_name(&self) -> &'static str {
        "Divergence check"
    }
}

#[cfg(test)]
mod tests {
    use crate::simulation::data::grid::multi_index::{CellIndex, FaceIndex, FromIndex};

use super::*;

    fn uniform_empty_grid() -> CPUSimulationData {
        let data = CPUSimulationData::new([3, 3]);
        data
    }

    #[test]
    fn test_divergence_computation_null() {
        let mut empty_grid = uniform_empty_grid();
        let mut test_operator = DivergenceComputation{};

        test_operator.execute(&mut empty_grid, &PhysicalConstants::default(), 1.0).unwrap();

        for d in empty_grid.divergence.data {
            assert_eq!(d, 0.0);
        }
    }

    #[test]
    fn test_divergence_computation() {
        let mut test_grid = uniform_empty_grid();
        test_grid.velocity[0].set(&FaceIndex::from_index([1, 1]), 1.0);
        test_grid.velocity[0].set(&FaceIndex::from_index([2, 1]), 1.0);
        let mut test_operator = DivergenceComputation{};

        test_operator.execute(&mut test_grid, &PhysicalConstants::default(), 1.0).unwrap();

        assert_eq!(*test_grid.divergence.get(&CellIndex::from_index([0, 1])), -1.0 / test_grid.grid.get_delta_x()[0]);
        assert_eq!(*test_grid.divergence.get(&CellIndex::from_index([1, 1])), 0.0);
        assert_eq!(*test_grid.divergence.get(&CellIndex::from_index([2, 1])), 1.0 / test_grid.grid.get_delta_x()[0]);
    }
}