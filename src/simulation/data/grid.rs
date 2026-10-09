use multi_index::{CellIndex, FaceIndex, FromIndex, TypedMultiIndex};

pub mod config;
pub mod multi_index;

#[derive(Clone)]
pub struct Grid<const D: usize> {
    dx: [f32; D],

    cell_dim: [usize; D],
    cell_index_stride: [usize; D],
}

impl<const D: usize> Grid<D> {
    pub fn new(dim: [usize; D], dx: [f32; D]) -> Self {
        let mut cell_index_stride = [1; D];
        for axis in 1..D {
            cell_index_stride[axis] = cell_index_stride[axis-1] * dim[axis-1];
        }

        Self {
            dx,
            cell_dim: dim,
            cell_index_stride,
        }
    }

    /// Iterates over all cells except the border cells
    pub fn cell_iter(&self) -> TypedMultiIndex<CellIndex<D>, D> {
        TypedMultiIndex::<CellIndex<D>, D>::new(&self.cell_dim)
            .exclude_border()
    }

    /// Iterates over all cells including the border cells
    pub fn total_cell_iter(&self) -> TypedMultiIndex<CellIndex<D>, D> {
        TypedMultiIndex::<CellIndex<D>, D>::new(&self.cell_dim)
    }

    /// Iterates over all faces of the specified axis excluding the border cells
    pub fn face_iter(&self, axis: usize) -> TypedMultiIndex<FaceIndex<D>, D> {
        TypedMultiIndex::<FaceIndex<D>,D>::new(&self.get_face_dimension(axis))
            .exclude_border()
    }

    /// Iterates over all faces of the specified axis including the border cells
    pub fn total_face_iter(&self, axis: usize) -> TypedMultiIndex<FaceIndex<D>, D> {
        TypedMultiIndex::<FaceIndex<D>,D>::new(&self.get_face_dimension(axis))
    }

    /// Returns a vector containing the succeeding cell in every direction.
    pub fn succeeding_cells(&self, cell_idx: &CellIndex<D>) -> Vec<(usize, CellIndex<D>)> {
        let mut result: Vec<(usize, CellIndex<D>)> = Vec::new();
        for axis in 0..D {
            let mut succeeding = cell_idx.idx;
            succeeding[axis] += 1;
            result.push((axis, CellIndex::from_index(succeeding)));
        }
        result
    }

    /// Indices of both faces in the specified axis
    pub fn get_face_indices(&self, cell_idx: &CellIndex<D>, axis: usize) -> [FaceIndex<D>; 2] {
        let minus_i_cell = cell_idx.idx;
        let mut plus_i_cell = minus_i_cell;
        plus_i_cell[axis] += 1;
        [
            FaceIndex::<D>::new(axis, minus_i_cell),
            FaceIndex::<D>::new(axis, plus_i_cell)
        ]
    }

    /// Indices of adjacent cells for a given face
    pub fn face_adjacent_cells(&self, face_idx: &FaceIndex<D>) -> [CellIndex<D>; 2] {
        let plus_i_cell = face_idx.idx;
        let mut minus_i_cell = plus_i_cell;
        minus_i_cell[face_idx.axis] -= 1;
        [
            CellIndex::<D>::from_index(minus_i_cell),
            CellIndex::<D>::from_index(plus_i_cell)
        ]
    }

    /// Index iterator over the cells at the domain boundary. Set start to true to select the boundary
    /// at coordinate 0.
    pub fn border_cell_iter(&self, axis: usize, start: bool) -> TypedMultiIndex<CellIndex<D>, D> {
        if start {
            TypedMultiIndex::<CellIndex<D>, D>::new(&self.cell_dim)
                .restrict_axis(axis, 0)
        } else {
            TypedMultiIndex::<CellIndex<D>, D>::new(&self.cell_dim)
                .restrict_axis(axis, self.cell_dim[axis]-1)
        }
    }

    /// Index iterator over the faces at the domain boundary. Set start to true to select the boundary
    /// at coordinate 0.
    pub fn border_face_iter(&self, axis: usize, start: bool) -> TypedMultiIndex<FaceIndex<D>, D> {
        if start {
            TypedMultiIndex::<FaceIndex<D>, D>::new(&self.get_face_dimension(axis))
                .set_upper_bound(axis, 1)
        } else {
            TypedMultiIndex::<FaceIndex<D>, D>::new(&self.get_face_dimension(axis))
                .set_lower_bound(axis, self.cell_dim[axis]-1)
        }
    }

    /// Index iterator over all domain boundary faces. More general case of border_face_iter.
    pub fn domain_boundary_face_iter(&self) -> Vec<(usize, TypedMultiIndex<FaceIndex<D>, D>)> {
        let mut result = Vec::<(usize, TypedMultiIndex<FaceIndex<D>, D>)>::new();
        result.push((0, self.border_face_iter(0, true)));
        result.push((0, self.border_face_iter(0, false)));
        result.push((1, self.border_face_iter(1, true)));
        result.push((1, self.border_face_iter(1, false)));
        result
    }

    /// Transforms the given cell index into a linear array index given the stored grid dimensions.
    pub fn linearize_cell_index(&self, cell_idx: &CellIndex<D>) -> usize {
        let mut sum = 0;
        for axis in 0..D {
            sum += cell_idx.idx[axis] * self.cell_index_stride[axis];
        }
        sum
    }

    /// Transforms the given face index into a linear array index given the stored grid dimensions.
    pub fn linearize_face_index(&self, face_idx: &FaceIndex<D>) -> usize {
        let axis = face_idx.axis;
        let dim = self.get_face_dimension(axis);
        face_idx.idx[0] + face_idx.idx[1] * dim[0]
    }

    pub fn get_total_cell_count(&self) -> usize {
        self.cell_dim.iter().product()
    }

    pub fn get_total_face_count(&self, axis: usize) -> usize {
        self.get_face_dimension(axis).iter().product()
    }

    pub fn get_outer_size(&self) -> [usize; D] {
        self.cell_dim
    }

    pub fn get_delta_x(&self, axis: usize) -> f32 {
        self.dx[axis]
    }

    pub fn get_face_dimension(&self, axis: usize) -> [usize; D] {
        let mut face_dimension = self.cell_dim;
        face_dimension[axis] += 1;
        face_dimension
    }

    pub fn face_world_position(
        &self,
        face: &FaceIndex<D>,
    ) -> [f32; D] {
        let mut result = [0.0f32; D];
        for i in 0..D {
            if face.axis == i {
                result[i] = (face.idx[i] as f32) * self.dx[i];
            } else {
                result[i] = (face.idx[i] as f32 + 0.5) * self.dx[i];
            }
        }
        result
    }

    pub fn cell_world_position(
        &self,
        cell: &CellIndex<D>,
    ) -> [f32; D] {
        let mut result = [0.0f32; D];
        for i in 0..D {
            result[i] = (cell.idx[i] as f32 + 0.5) * self.dx[i];
        }
        result
    }
}

pub type Grid2D = Grid<2>;

#[cfg(test)]
mod tests {
    use super::*;

    fn test_grid() -> Grid2D {
        Grid2D::new(
            [20, 10],
            [0.05, 0.1]
        )
    }

    #[test]
    fn test_grid_construction() {
        let test_grid = test_grid();
        assert_eq!(test_grid.get_total_cell_count(), 200);
        assert_eq!(test_grid.get_total_face_count(0), 210);
    }

    #[test]
    fn test_grid_indexing() {
        let test_grid = test_grid();

        for (lin_idx, cell_idx) in test_grid.total_cell_iter().enumerate() {
            let linearized = test_grid.linearize_cell_index(&cell_idx);
            println!("{}: {:?} -> {}", lin_idx, cell_idx.idx, linearized);

            assert_eq!(lin_idx, linearized);
        }
    }

    #[test]
    fn test_border_iter() {
        let test_grid = test_grid();

        let mut cell_sum = 0;
        for cell_idx in test_grid.border_cell_iter(0, true) {
            assert_eq!(cell_idx.idx[0], 0);
            cell_sum += 1;
        }
        assert_eq!(cell_sum, 10);
    }
}