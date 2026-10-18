use multi_index::{CellIndex, FaceIndex, FromIndex, TypedMultiIndex};

use crate::simulation::data::grid::multi_index::FaceIndexIter;

pub mod config;
pub mod multi_index;
pub mod layout;

#[derive(Clone)]
pub struct Grid<const D: usize> {
    dx: [f32; D],

    dimensions: [usize; D],
}

impl<const D: usize> Grid<D> {
    pub fn new(dim: [usize; D], dx: [f32; D]) -> Self {
        Self {
            dx,
            dimensions: dim,
        }
    }

    /// Iterates over all cells
    pub fn all_cells(&self) -> TypedMultiIndex<CellIndex<D>, D> {
        TypedMultiIndex::<CellIndex<D>, D>::new(&self.dimensions)
    }

    /// Iterates over all cells except the border cells
    pub fn interior_cells(&self) -> TypedMultiIndex<CellIndex<D>, D> {
        TypedMultiIndex::<CellIndex<D>, D>::new(&self.dimensions)
            .exclude_border()
    }

    /// Iterates over all faces of the specified axis
    pub fn all_faces(&self, axis: usize) -> FaceIndexIter<D> {
        FaceIndexIter::<D>::new(self.dimensions, axis)
    }

    /// Iterates over all faces of the specified axis excluding the border cells
    pub fn interior_faces(&self, axis: usize) -> FaceIndexIter<D> {
        FaceIndexIter::<D>::new(self.dimensions, axis)
            .set_lower_bound(axis, 1)
            .set_upper_bound(axis, self.dimensions[axis]-1)
    }

    /// Returns a vector containing the succeeding cell and their axis in every direction.
    pub fn succeeding_cells(&self, cell_idx: &CellIndex<D>) -> Vec<(usize, CellIndex<D>)> {
        let mut result: Vec<(usize, CellIndex<D>)> = Vec::new();
        for axis in 0..D {
            let mut succeeding = cell_idx.idx;
            succeeding[axis] += 1;
            if succeeding[axis] < self.dimensions[axis] {
                result.push((axis, CellIndex::from_index(succeeding)));
            }
        }
        result
    }

    /// Returns a vector containing the preceeding cell and their axis in every direction.
    pub fn preceeding_cells(&self, cell_idx: &CellIndex<D>) -> Vec<(usize, CellIndex<D>)> {
        let mut result: Vec<(usize, CellIndex<D>)> = Vec::new();
        for axis in 0..D {
            let mut preceeding = cell_idx.idx;
            preceeding[axis] = preceeding[axis].wrapping_sub(1);
            if preceeding[axis] < self.dimensions[axis] {
                result.push((axis, CellIndex::from_index(preceeding)));
            }
        }
        result
    }

    /// Indices of both faces in the specified axis
    pub fn get_face_indices(&self, cell_idx: &CellIndex<D>, axis: usize) -> [FaceIndex<D>; 2] {
        let minus_i_face = cell_idx.idx;
        let mut plus_i_face = minus_i_face;
        plus_i_face[axis] += 1;
        [
            FaceIndex::<D>::new(axis, minus_i_face),
            FaceIndex::<D>::new(axis, plus_i_face)
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
            TypedMultiIndex::<CellIndex<D>, D>::new(&self.dimensions)
                .restrict_axis(axis, 0)
        } else {
            TypedMultiIndex::<CellIndex<D>, D>::new(&self.dimensions)
                .restrict_axis(axis, self.dimensions[axis]-1)
        }
    }

    /// Index iterator over the faces at the domain boundary. Set start to true to select the boundary
    /// at coordinate 0.
    pub fn border_face_iter(&self, axis: usize, start: bool) -> FaceIndexIter<D> {
        if start {
            FaceIndexIter::<D>::new(self.dimensions, axis)
                .restrict_axis(axis, 0)
        } else {
            FaceIndexIter::<D>::new(self.dimensions, axis)
                .restrict_axis(axis, self.dimensions[axis])
        }
    }

    /// Index iterator over all domain boundary faces. More general case of border_face_iter.
    pub fn domain_boundary_face_iter(&self) -> Vec<(usize, FaceIndexIter<D>)> {
        let mut result = Vec::<(usize, FaceIndexIter<D>)>::new();
        result.push((0, self.border_face_iter(0, true)));
        result.push((0, self.border_face_iter(0, false)));
        result.push((1, self.border_face_iter(1, true)));
        result.push((1, self.border_face_iter(1, false)));
        result
    }

    pub fn get_delta_x(&self) -> [f32; D] {
        self.dx
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
        assert_eq!(test_grid.dimensions.iter().product::<usize>(), 200);
        let mut face_d = test_grid.dimensions;
        face_d[0] += 1;
        assert_eq!(face_d.iter().product::<usize>(), 210);
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