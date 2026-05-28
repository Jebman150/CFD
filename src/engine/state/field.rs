use std::ops::Mul;

use crate::engine::grid::{Grid2D, multi_index::{CellIndex, FaceIndex, FromIndex}};

#[derive(Clone, Copy, PartialEq)]
pub enum CellType {
    Solid,
    Fluid
}

#[derive(Clone)]
pub struct Field<T, S> {
    pub data: Vec<T>,
    _marker: std::marker::PhantomData<S>
}

impl<S> Field<f32, S> {
    pub fn new(size: usize) -> Self {
        Field {
            data: vec![0f32; size],
            _marker: std::marker::PhantomData
        }
    }
    
    pub fn print(&self) {
        println!("{:?}", self.data);
    }
}

impl<S> Field<CellType, S> {
    pub fn new(size: usize) -> Self {
        Field {
            data: vec![CellType::Fluid; size],
            _marker: std::marker::PhantomData
        }
    }
}

impl<T> Field<T, CellIndex<2>> 
where
    T: Copy
{
    pub fn get(
        &self,
        grid: &Grid2D,
        idx: &CellIndex<2>
    ) -> &T {
        &self.data[grid.linearize_cell_index(idx)]
    }

    pub fn set(
        &mut self,
        grid: &Grid2D,
        idx: &CellIndex<2>,
        value: T
    ) {
        self.data[grid.linearize_cell_index(idx)] = value;
    }
}

impl Field<f32, CellIndex<2>>
{
    pub fn sample(
        &self,
        grid: &Grid2D,
        pos: [f32; 2],
    ) -> f32 {
        let dx = grid.get_delta_x();

        let gx = pos[0] / dx - 0.5;
        let gy = pos[1] / dx - 0.5;

        let i0 = gx.floor() as isize;
        let j0 = gy.floor() as isize;

        let tx = gx - i0 as f32;
        let ty = gy - j0 as f32;

        let i1 = i0 + 1;
        let j1 = j0 + 1;

        let v00 = self.sample_safe(
            grid,
            i0,
            j0,
        );

        let v10 = self.sample_safe(
            grid,
            i1,
            j0,
        );

        let v01 = self.sample_safe(
            grid,
            i0,
            j1,
        );

        let v11 = self.sample_safe(
            grid,
            i1,
            j1,
        );

        let vx0 =
            (1.0 - tx) * v00
            + tx * v10;

        let vx1 =
            (1.0 - tx) * v01
            + tx * v11;

        (1.0 - ty) * vx0
            + ty * vx1
    }

    fn sample_safe(
        &self,
        grid: &Grid2D,
        i: isize,
        j: isize,
    ) -> &f32 {
        let dims = grid.get_outer_size();

        let ii = i.clamp(0, dims[0] as isize - 1) as usize;

        let jj = j.clamp(0, dims[1] as isize - 1) as usize;

        let cell =
            CellIndex::<2>::from_index(
                [ii, jj],
            );

        &self.data[grid.linearize_cell_index(&cell)]
    }
}

impl<T> Field<T, FaceIndex<2>> 
where
    T: Copy
{
    pub fn get(
        &self,
        grid: &Grid2D,
        idx: &FaceIndex<2>
    ) -> &T {
        &self.data[grid.linearize_face_index(idx)]
    }

    pub fn set(
        &mut self,
        grid: &Grid2D,
        idx: &FaceIndex<2>,
        value: T
    ) {
        self.data[grid.linearize_face_index(idx)] = value;
    }
}

impl Field<f32, FaceIndex<2>>
{
    pub fn sample(
        &self,
        grid: &Grid2D,
        axis: usize,
        pos: [f32; 2],
    ) -> f32 {
        let dx = grid.get_delta_x();

        /*
            Convert world position to local MAC coordinates.
        */

        let offset = if axis == 0 {
            [0.0, 0.5]
        } else {
            [0.5, 0.0]
        };

        let gx = pos[0] / dx - offset[0];
        let gy = pos[1] / dx - offset[1];

        let i0 = gx.floor() as isize;
        let j0 = gy.floor() as isize;

        let tx = gx - i0 as f32;
        let ty = gy - j0 as f32;

        let i1 = i0 + 1;
        let j1 = j0 + 1;

        let v00 = self.sample_safe(
            grid,
            axis,
            i0,
            j0,
        );

        let v10 = self.sample_safe(
            grid,
            axis,
            i1,
            j0,
        );

        let v01 = self.sample_safe(
            grid,
            axis,
            i0,
            j1,
        );

        let v11 = self.sample_safe(
            grid,
            axis,
            i1,
            j1,
        );

        let vx0 =
            (1.0 - tx) * v00
            + tx * v10;

        let vx1 =
            (1.0 - tx) * v01
            + tx * v11;

        (1.0 - ty) * vx0
            + ty * vx1
    }

    fn sample_safe(
        &self,
        grid: &Grid2D,
        axis: usize,
        i: isize,
        j: isize,
    ) -> &f32 {

        let dims =
            grid.get_mac_dimension(axis);

        let ii = i.clamp(0, dims[0] as isize - 1)
            as usize;

        let jj = j.clamp(0, dims[1] as isize - 1)
            as usize;

        let face =
            FaceIndex::<2>::new(
                axis,
                [ii, jj],
            );

        &self.data[grid.linearize_face_index(&face)]
    }
}