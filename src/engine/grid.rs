use crate::engine::grid::multi_index::{CellIndex, FaceIndex, FromIndex, TypedMultiIndex};

mod config;
pub mod multi_index;



enum FaceType {
    SolidInternal,
    Solid,
    Fluid,
}

pub struct Grid2D {
    boundary_condition: Vec<config::BoundaryCondition>,

    dx: f32,
    outer_dim: [usize; 2],
    inner_dim: [usize; 2],

    inner_index_stride: [usize; 2],
    outer_index_stride: [usize; 2],
}

impl Grid2D {
    pub fn new(source_file: String) -> Self {
        let config = config::Config::from_file(&source_file).unwrap();

        println!("Grid initialization: ");
        println!(" | Grid size: {:?}", config.grid_size);
        println!(" | dx: {:?}", config.delta_x);

        let inner_dimension: [usize; 2] = [config.grid_size[0], config.grid_size[1]];
        let outer_dimension: [usize; 2] = [inner_dimension[0]+2, inner_dimension[1]+2];

        let inner_stride = [1, inner_dimension[0]];
        let outer_stride = [1, outer_dimension[0]];

        Self {
            boundary_condition: config.boundary_condition,

            dx: config.delta_x,
            outer_dim: outer_dimension,
            inner_dim: inner_dimension,

            inner_index_stride: inner_stride,
            outer_index_stride: outer_stride
        }
    }

    pub fn cell_iter(&self) -> TypedMultiIndex<CellIndex<2>, 2> {
        TypedMultiIndex::<CellIndex<2>, 2>::new(&self.outer_dim)
            .exclude_border()
    }

    pub fn total_cell_iter(&self) -> TypedMultiIndex<CellIndex<2>, 2> {
        TypedMultiIndex::<CellIndex<2>, 2>::new(&self.outer_dim)
    }

    pub fn face_iter(&self, axis: usize) -> TypedMultiIndex<FaceIndex<2>, 2> {
        TypedMultiIndex::<FaceIndex<2>,2>::new(&self.get_mac_dimension(axis))
            .exclude_border()
    }

    pub fn total_face_iter(&self, axis: usize) -> TypedMultiIndex<FaceIndex<2>, 2> {
        TypedMultiIndex::<FaceIndex<2>,2>::new(&self.get_mac_dimension(axis))
    }

    pub fn succeeding_cells(&self, cell_idx: &CellIndex<2>) -> Vec<(usize, CellIndex<2>)> {
        let mut result: Vec<(usize, CellIndex<2>)> = Vec::new();
        for axis in 0..2 {
            let mut succeeding = cell_idx.idx;
            succeeding[axis] += 1;
            result.push((axis, CellIndex::from_index(succeeding)));
        }
        result
    }

    pub fn cell_faces(&self, cell_idx: &CellIndex<2>, axis: usize) -> [FaceIndex<2>; 2] {
        let mut opposite_face = cell_idx.idx;
        opposite_face[axis] += 1;
        [
            FaceIndex::<2>::new(axis, cell_idx.idx),
            FaceIndex::<2>::new(axis, opposite_face)
        ]
    }

    pub fn face_adjacent_cells(&self, face_idx: &FaceIndex<2>) -> [CellIndex<2>; 2] {
        let mut opposite_cell = face_idx.idx;
        opposite_cell[face_idx.axis] -= 1;
        [
            CellIndex::<2>::from_index(opposite_cell),
            CellIndex::<2>::from_index(face_idx.idx)
        ]
    }

    pub fn get_face_indices(&self, cell_idx: &CellIndex<2>, axis: usize) -> [FaceIndex<2>; 2] {
        let mut indices = cell_idx.idx;
        indices[axis] += 1;
        [FaceIndex::<2>::new(axis, cell_idx.idx), FaceIndex::<2>::new(axis, indices)]
    }

    pub fn border_cell_iter(&self, axis: usize, start: bool) -> TypedMultiIndex<CellIndex<2>, 2> {
        if start {
            TypedMultiIndex::<CellIndex<2>, 2>::new(&self.outer_dim)
                .restrict_axis(axis, 0)
        } else {
            TypedMultiIndex::<CellIndex<2>, 2>::new(&self.outer_dim)
                .restrict_axis(axis, self.outer_dim[axis]-1)
        }
    }

    pub fn solid_border_iter(&self) -> Vec<TypedMultiIndex<CellIndex<2>, 2>> {
        let mut result = Vec::<TypedMultiIndex::<CellIndex<2>, 2>>::new();
        if self.boundary_condition[0].value == 0.0 {
            result.push(self.border_cell_iter(0, true));
        }
        if self.boundary_condition[1].value == 0.0 {
            result.push(self.border_cell_iter(0, false));
        }
        if self.boundary_condition[2].value == 0.0 {
            result.push(self.border_cell_iter(1, true));
        }
        if self.boundary_condition[3].value == 0.0 {
            result.push(self.border_cell_iter(1, false));
        }
        result
    }

    pub fn border_face_iter(&self, axis: usize, start: bool) -> TypedMultiIndex<FaceIndex<2>, 2> {
        if start {
            TypedMultiIndex::<FaceIndex<2>, 2>::new(&self.get_mac_dimension(axis))
                .set_upper_bound(axis, 1)
        } else {
            TypedMultiIndex::<FaceIndex<2>, 2>::new(&self.get_mac_dimension(axis))
                .set_lower_bound(axis, self.outer_dim[axis]-1)
        }
    }

    pub fn domain_boundary_face_iter(&self) -> Vec<(usize, TypedMultiIndex<FaceIndex<2>, 2>)> {
        let mut result = Vec::<(usize, TypedMultiIndex<FaceIndex<2>, 2>)>::new();
        result.push((0, self.border_face_iter(0, true)));
        result.push((0, self.border_face_iter(0, false)));
        result.push((1, self.border_face_iter(1, true)));
        result.push((1, self.border_face_iter(1, false)));
        result
    }
    
    pub fn get_condition(&self, axis: usize, start: bool) -> f32 {
        if start {
            self.boundary_condition[axis*2].value
        } else {
            self.boundary_condition[axis*2+1].value
        }
    }

    pub fn linearize_cell_index(&self, cell_idx: &CellIndex<2>) -> usize {
        cell_idx.idx[0] * self.outer_index_stride[0] + cell_idx.idx[1] * self.outer_index_stride[1]
    }

    pub fn linearize_inner_cell_index(&self, cell_idx: &CellIndex<2>) -> usize {
        (cell_idx.idx[0]-1) * self.inner_index_stride[0] + (cell_idx.idx[1]-1) * self.inner_index_stride[1]
    }

    pub fn linearize_face_index(&self, face_idx: &FaceIndex<2>) -> usize {
        let axis = face_idx.axis;
        let dim = self.get_mac_dimension(axis);
        face_idx.idx[0] + face_idx.idx[1] * dim[0]
    }

    pub fn get_total_cell_count(&self) -> usize {
        self.outer_dim.iter().product()
    }

    pub fn get_cell_count(&self) -> usize {
        self.inner_dim.iter().product()
    }

    pub fn get_total_face_count(&self, axis: usize) -> usize {
        self.get_mac_dimension(axis).iter().product()
    }

    pub fn get_size(&self, axis: usize) -> usize {
        self.inner_dim[axis]
    }

    pub fn get_outer_size(&self) -> [usize; 2] {
        self.outer_dim
    }

    pub fn get_delta_x(&self) -> f32 {
        self.dx
    }

    pub fn get_mac_dimension(&self, axis: usize) -> [usize; 2] {
        let mut mac_dimension = self.outer_dim;
        mac_dimension[axis] += 1;
        mac_dimension
    }

    pub fn face_world_position(
        &self,
        face: &FaceIndex<2>,
    ) -> [f32; 2] {
        let mut pos = [
            face.idx[0] as f32 * self.dx,
            face.idx[1] as f32 * self.dx,
        ];

        if face.axis == 0 {
            pos[1] += 0.5 * self.dx;
        } else {
            pos[0] += 0.5 * self.dx;
        }
        pos
    }

    pub fn cell_world_position(
        &self,
        cell: &CellIndex<2>,
    ) -> [f32; 2] {
        [
            cell.idx[0] as f32 * self.dx + 0.5*self.dx,
            cell.idx[1] as f32 * self.dx + 0.5*self.dx,
        ]
    }

    /*

    pub fn get_divergence(&self) -> Vec<f32> {
        let mut result = Vec::<f32>::new();
        for idx in self.get_cell_iterator() {
            let mut div = 0f32;
            for (axis, vel_field) in self.velocities.iter().enumerate() {
                let mut opposite_face = idx.clone();
                opposite_face[axis] += 1;
                div += (vel_field.get(&opposite_face) - vel_field.get(&idx)) / self.dx;
            }
            result.push(div);
        }
        result
    }

    pub fn apply_boundary_condition(&mut self) {
        for (axis, vel_field) in self.velocities.iter_mut().enumerate() {
            let bc = self.boundary_condition[axis*2].value;
            vel_field.set_in_plane(axis, 0, bc);
            let bc = self.boundary_condition[axis*2+1].value;
            vel_field.set_in_plane(axis, vel_field.get_size(axis)-1, bc);
        }
    }

    pub fn apply_solid_cells(&mut self) {
        for idx in self.get_cell_iterator() {
            if self.cell_types.get(&idx) == CellType::Solid {
                self.set_cell_faces(&idx, 0f32);
            }
        }
    }

    pub fn set_border_solid(&mut self) {
        for axis in self.get_axes() {
            if self.boundary_condition[axis*2].value == 0f32 {
                self.cell_types.set_in_plane(axis, 0, CellType::Solid);
            }
            if self.boundary_condition[axis*2+1].value == 0f32 {
                self.cell_types.set_in_plane(axis, self.cell_types.get_size(axis)-1, CellType::Solid);
            }
        }
    }

    pub fn set_cell_faces(&mut self, idx: &Vec<usize>, val: f32) {
        for (axis, vel_field) in self.velocities.iter_mut().enumerate() {
            vel_field.set(&idx, 0f32);

            let mut opposite_face = idx.clone();
            opposite_face[axis] += 1;
            vel_field.set(&opposite_face, 0f32);
        }
    }

    pub fn is_boundary_cell(&self, idx: &Vec<usize>) -> bool {
        for axis in self.get_axes() {
            if idx[axis] >= self.inner_dim[axis] {
                return true;
            }
        }
        return false;
    }

    pub fn get_cell_type(&self, idx: &Vec<usize>) -> CellType {
        self.cell_types.get(&idx.iter().map(|i| i + 1).collect())
    }

    pub fn get_succeeding_cell_indices(&self, idx: &Vec<usize>) -> Vec<(usize, Vec<usize>)> {
        let mut result: Vec<(usize, Vec<usize>)> = Vec::new();
        for axis in self.get_axes() {
            let mut succeeding = idx.clone();
            succeeding[axis] += 1;
            result.push((axis, succeeding));
        }
        result
    }

    pub fn cell_iterator(&self) -> CellIndex<2> {
        CellIndex {
            indices: vec![1, 1],
            lower: vec![1, 1],
            upper: vec![self.outer_dim[0], self.outer_dim[1]],
            done: false
        }
    }

    pub fn get_cell_iterator(&self) -> MultiIndex {
        CartesianProduct::new(&self.get_dimensions()).into_iter()
    }

    pub fn get_inner_iterator(&self) -> MultiIndex {
        CartesianProduct::new(&self.get_dimensions()).into_iter()
    }

    pub fn get_delta_x(&self) -> f32 {
        self.dx
    }

    pub fn to_cell_index(&self, idx: &Vec<usize>) -> usize {
        idx.iter().zip(self.inner_index_stride.iter()).map(|(idx, stride)| idx * stride).sum()
    }

    pub fn get_velocities(&self, axis: usize) -> Vec<f32> {
        self.velocities[axis].get_data()
    }

    pub fn get_velocity_field(&self, axis: usize) -> Field<f32> {
        self.velocities[axis].clone()
    }

    pub fn get_outer_cell_count(&self) -> usize {
        self.get_outer_dimensions().into_iter().product()
    }

    pub fn get_cell_count(&self) -> usize {
        self.get_dimensions().into_iter().product()
    }

    pub fn get_size(&self, axis: usize) -> usize {
        self.inner_dim[axis]
    }

    pub fn get_outer_dimensions(&self) -> Vec<usize> {
        self.outer_dim.clone()
    }

    pub fn get_dimensions(&self) -> Vec<usize> {
        self.inner_dim.clone()
    }

    pub fn get_axes(&self) -> Vec<usize> {
        (0..(self.velocities.len())).collect()
    }*/
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_construction() {
        let test_grid = Grid2D::new("config/test/grid_topology.toml".to_string());
        assert_eq!(test_grid.get_total_cell_count(), 264);
        assert_eq!(test_grid.get_total_face_count(0), 276);
    }

    #[test]
    fn test_grid_indexing() {
        let test_grid = Grid2D::new("config/test/grid_topology.toml".to_string());
        
        let mut index_counter = 0;
        let mut offset_multiplier = 1;
        let offset = 2;

        for (lin_idx, cell_idx) in test_grid.cell_iter().enumerate() {
            let linearized = test_grid.linearize_cell_index(&cell_idx);
            let expected = lin_idx + offset * offset_multiplier + 21;
            println!("{}: {:?} -> {}", expected, cell_idx.idx, linearized);

            assert!(cell_idx.idx[0] > 0 && cell_idx.idx[1] > 0);
            assert_eq!(expected, linearized);

            index_counter += 1;
            if index_counter == 20 {
                offset_multiplier += 1;
                index_counter = 0;
            }
        }

        for (lin_idx, cell_idx) in test_grid.cell_iter().enumerate() {
            let linearized = test_grid.linearize_inner_cell_index(&cell_idx);
            println!("{}: {:?} -> {}", lin_idx, cell_idx.idx, linearized);

            assert_eq!(lin_idx, linearized);
        }
    }

    #[test]
    fn test_border_iter() {
        let test_grid = Grid2D::new("config/test/grid_topology.toml".to_string());

        let mut cell_sum = 0;
        for cell_idx in test_grid.border_cell_iter(0, true) {
            assert_eq!(cell_idx.idx[0], 0);
            cell_sum += 1;
        }
        assert_eq!(cell_sum, 12);

        cell_sum = 0;
        let iter = test_grid.solid_border_iter();
        assert_eq!(iter.len(), 4);
        for border_iter in iter {
            for cell_idx in border_iter {
                assert!(
                    cell_idx.idx[0] == 0 ||
                    cell_idx.idx[0] == 21 ||
                    cell_idx.idx[1] == 0 ||
                    cell_idx.idx[1] == 11
                );
                cell_sum += 1;
            }
        }
        assert_eq!(cell_sum, 12*2 + 22*2);

    }
}