use std::time::Instant;

use crate::simulation::{PhysicalConstants, data::{SimulationData, grid::{config, layout::GridLayout, multi_index::{CellIndex, FaceIndex, FromIndex}}}, pipeline::operator::SimulationOperator};



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
        _physical_constants: &PhysicalConstants,
        _dt: f32,
    ) -> Result<u128, String> {
        let start = Instant::now();
        puffin::profile_scope!("apply_domain_boundary_condition");
        for mut face_idx in data.grid.border_face_iter(0, true) {
            face_idx.axis = 0;
            data.velocity[0].set(&face_idx, self.value_left);
        }
        for mut face_idx in data.grid.border_face_iter(0, false) {
            face_idx.axis = 0;
            data.velocity[0].set(&face_idx, self.value_right);
        }
        for mut face_idx in data.grid.border_face_iter(1, true) {
            face_idx.axis = 1;
            data.velocity[1].set(&face_idx, self.value_top);
        }
        for mut face_idx in data.grid.border_face_iter(1, false) {
            face_idx.axis = 1;
            data.velocity[1].set(&face_idx, self.value_bottom);
        }
        Ok(start.elapsed().as_micros())
    }

    fn get_name(&self) -> &'static str {
        "Apply domain boundary condition"
    }
}


pub struct ApplyInteriorBoundaryCondition;

impl ApplyInteriorBoundaryCondition {

    fn apply_internal_velocity(&self, data: &mut SimulationData) {
        return;
        let mut face_index = FaceIndex::from_index([
            data.pressure.layout.dimensions()[0] / 2,
            data.pressure.layout.dimensions()[1] / 2
        ]);
        let range = face_index.idx[0] / 4;
        face_index.axis = 0;

        for x in (face_index.idx[0] - range)..(face_index.idx[0] + range) {
            data.velocity[0].set(&FaceIndex::from_index([x, face_index.idx[1]]), 1.0);
        }
        data.velocity[0].set(&FaceIndex::from_index(face_index.idx), 1.0);
    }

    
    fn apply_solid_cells_condition(&self, data: &mut SimulationData) {
        for cell_idx in data.grid.all_cells() {
            if *data.is_solid.get(&cell_idx) {
                self.set_cell_faces(data, &cell_idx, 0.0);
            }
        }
    }

    fn set_cell_faces(&self, data: &mut SimulationData, cell_idx: &CellIndex<2>, value: f32) {
        for axis in 0..2 {
            for face_idx in data.grid.get_face_indices(cell_idx, axis) {
                data.velocity[axis].set(&face_idx, value);
            }
        }
    }

    pub fn set_solid_border(data: &mut SimulationData) {
        for cell_idx in data.grid
            .border_cell_iter(1, true)
            .chain(data.grid.border_cell_iter(1, false))
        {
            data.is_solid.set(&cell_idx,  true);
        }
    }

    pub fn set_solid_obstacle(data: &mut SimulationData) {
        let mut position: [usize; 2] = data.is_solid.layout.dimensions();
        position[0] = position[0] / 4;
        position[1] = position[1] / 2;
        let radius = (data.is_solid.layout.dimensions()[0] / 16) as i32;

        for cell_idx in data.grid.interior_cells() {
            if (position[0] as i32 - cell_idx.idx[0] as i32).abs() < radius &&
                (position[1] as i32 - cell_idx.idx[1] as i32).abs() < radius
            {
                data.is_solid.set(&cell_idx, true);
            }
        }
    }

    /// flow_edge_allowed bit mask: Left, Top, Right, Bottom
    pub fn precompute_edge_flow_from_solid_cells(data: &mut SimulationData) {
        for cell_idx in data.grid.all_cells() {
            if *data.is_solid.get(&cell_idx) {
                continue;
            }
            
            for (axis, neighbour_index) in data.grid.succeeding_cells(&cell_idx).iter()
            {
                data.flow_edge_count[*axis].set(neighbour_index, data.flow_edge_count[*axis].get(neighbour_index) + 1);
                data.flow_edge_allowed.set(neighbour_index, data.flow_edge_allowed.get(neighbour_index) | (1 << axis));
            }
            for (axis, neighbour_index) in data.grid.preceeding_cells(&cell_idx).iter()
            {
                data.flow_edge_count[*axis].set(neighbour_index, data.flow_edge_count[*axis].get(neighbour_index) + 1);
                data.flow_edge_allowed.set(neighbour_index, data.flow_edge_allowed.get(neighbour_index) | (1 << (axis + 2)));
            }
        }
    }
}

impl SimulationOperator for ApplyInteriorBoundaryCondition {
    fn execute(
        &mut self,
        data: &mut SimulationData,
        _physical_constants: &PhysicalConstants,
        _dt: f32,
    ) -> Result<u128, String> {
        let start = Instant::now();
        puffin::profile_scope!("apply_internal_boundary_condition");
        self.apply_solid_cells_condition(data);
        self.apply_internal_velocity(data);
        Ok(start.elapsed().as_micros())
    }
    
    fn get_name(&self) -> &'static str {
        "Apply interior boundary condition"
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    fn uniform_empty_grid() -> SimulationData {
        let data = SimulationData::new([20, 10]);
        data
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
        let mut test_grid = uniform_empty_grid();
        let mut test_condition = left_in_top_out();

        test_condition.execute(&mut test_grid, &PhysicalConstants::default(), 1.0).unwrap();

        for mut face_idx in test_grid.grid.border_face_iter(0, true) {
            face_idx.axis = 0;
            assert_eq!(test_grid.velocity[0].get(&face_idx), 1.0f32);
        }
        for mut face_idx in test_grid.grid.border_face_iter(1, true) {
            face_idx.axis = 1;
            assert_eq!(test_grid.velocity[1].get(&face_idx), -1.0f32);
        }
    }
}