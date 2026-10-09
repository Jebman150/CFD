use std::time::Instant;

use crate::simulation::{data::{SimulationData, grid::{Grid2D, config, multi_index::{CellIndex, FaceIndex, FromIndex}}}, pipeline::operator::SimulationOperator};



pub struct ApplyDomainBoundaryCondition {
    value_left: f32,
    value_right: f32,
    value_top: f32,
    value_bottom: f32,
}

impl ApplyDomainBoundaryCondition {
    pub fn from_config(source_file: &str) -> Self {
        let config = config::Config::from_file(source_file).unwrap();
        Self {
            value_left: config.boundary_condition[0].value,
            value_right: config.boundary_condition[1].value,
            value_top: config.boundary_condition[2].value,
            value_bottom: config.boundary_condition[3].value,
        }
    }
}

impl SimulationOperator for ApplyDomainBoundaryCondition {
    fn execute(
        &mut self,
        data: &mut SimulationData,
        context: &mut Grid2D,
    ) -> Result<u128, String> {
        let start = Instant::now();
        for mut face_idx in context.border_face_iter(0, true) {
            face_idx.axis = 0;
            data.velocity[0].set(&context, &face_idx, self.value_left);
        }
        for mut face_idx in context.border_face_iter(0, false) {
            face_idx.axis = 0;
            data.velocity[0].set(&context, &face_idx, self.value_right);
        }
        for mut face_idx in context.border_face_iter(1, true) {
            face_idx.axis = 1;
            data.velocity[1].set(&context, &face_idx, self.value_top);
        }
        for mut face_idx in context.border_face_iter(1, false) {
            face_idx.axis = 1;
            data.velocity[1].set(&context, &face_idx, self.value_bottom);
        }
        Ok(start.elapsed().as_micros())
    }

    fn get_name(&self) -> &'static str {
        "Apply domain boundary condition"
    }
}


pub struct ApplyInteriorBoundaryCondition;

impl ApplyInteriorBoundaryCondition {

    fn apply_internal_velocity(&self, data: &mut SimulationData, context: &Grid2D) {
        let mut face_index = FaceIndex::from_index([50, 50]);
        face_index.axis = 0;

        data.velocity[0].set(context, &face_index, 0.1);
    }

    /*
    fn apply_solid_cells_condition(&self, data: &mut SimulationData, context: &Grid2D) {
        for cell_idx in context.total_cell_iter() {
            if *data.cell_types.get(&context, &cell_idx) == CellType::Solid {
                self.set_cell_faces(data, context, &cell_idx, 0.0);
            }
        }
    }

    fn set_cell_faces(&self, data: &mut SimulationData, context: &Grid2D, cell_idx: &CellIndex<2>, value: f32) {
        for axis in 0..2 {
            for face_idx in context.get_face_indices(cell_idx, axis) {
                data.velocity[axis].set(&context, &face_idx, value);
            }
        }
    }

    fn set_solid_border(&self, data: &mut SimulationData,  context: &Grid2D) {
        for border_iter in context.solid_border_iter() {
            for cell_idx in border_iter {
                data.cell_types.set(&context, &cell_idx,  CellType::Solid);
            }
        }
    }

    fn set_solid_obstacle(&self, state: &mut SimulationData,  context: &Grid2D) {
        for cell_idx in context.cell_iter() {
            if cell_idx.idx[0] < (context.get_outer_size()[0] as f32 * 2.0 / 16.0) as usize ||
                cell_idx.idx[0] >= (context.get_outer_size()[0] as f32 * 4.0 / 16.0) as usize ||
                cell_idx.idx[1] < (context.get_outer_size()[1] as f32 * 7.0 / 16.0) as usize ||
                cell_idx.idx[1] >= (context.get_outer_size()[1] as f32 * 9.0 / 16.0) as usize
            {
                continue;
            }

            state.cell_types.set(&context, &cell_idx, CellType::Solid);
        }
    }*/
}

impl SimulationOperator for ApplyInteriorBoundaryCondition {
    fn execute(
        &mut self,
        data: &mut SimulationData,
        context: &mut Grid2D
    ) -> Result<u128, String> {
        let start = Instant::now();
        //self.apply_solid_cells_condition(data, context);
        self.apply_internal_velocity(data, context);
        Ok(start.elapsed().as_micros())
    }
    
    fn get_name(&self) -> &'static str {
        "Apply interior boundary condition"
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

    fn left_in_top_out() -> ApplyDomainBoundaryCondition {
        ApplyDomainBoundaryCondition {
            value_left: 1.0,
            value_right: 0.0,
            value_top: -1.0,
            value_bottom: 0.0
        }
    }

    #[test]
    fn test_boundary_condition_applies_correctly() {
        let (mut test_grid, mut context) = uniform_empty_grid();
        let mut test_condition = left_in_top_out();

        test_condition.execute(&mut test_grid, &mut context).unwrap();

        for mut face_idx in context.border_face_iter(0, true) {
            face_idx.axis = 0;
            assert_eq!(*test_grid.velocity[0].get(&context, &face_idx), 1.0f32);
        }
        for mut face_idx in context.border_face_iter(1, true) {
            face_idx.axis = 1;
            assert_eq!(*test_grid.velocity[1].get(&context, &face_idx), -1.0f32);
        }
    }
}