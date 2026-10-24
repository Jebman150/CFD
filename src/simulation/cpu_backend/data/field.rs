use crate::simulation::data::grid::{
    multi_index::{
        CellIndex,
        FaceIndex,
        FromIndex
    },
    layout::{
        CellLayout,
        FaceLayout,
        GridLayout,
    }
};

pub type CellField2<T> = Field<T, CellIndex<2>, CellLayout<2>>;
pub type CellField2F = Field<f32, CellIndex<2>, CellLayout<2>>;
pub type FaceField2F = Field<f32, FaceIndex<2>, FaceLayout<2>>;

#[derive(Clone)]
pub struct Field<T, S, L>
where
    L: GridLayout<2>
{
    pub data: Vec<T>,
    _marker: std::marker::PhantomData<S>,
    pub layout: L,
}

impl<T, S, L> Field<T, S, L>
where
    L: GridLayout<2> + Clone,
    T: Default + Clone
{
    pub fn new(layout: &L) -> Self {
        Self {
            data: vec![T::default(); layout.total_count()],
            _marker: std::marker::PhantomData,
            layout: layout.clone()
        }
    }
}

impl<T> Field<T, CellIndex<2>, CellLayout<2>> 
where
    T: Copy
{
    pub fn get(
        &self,
        idx: &CellIndex<2>
    ) -> &T {
        &self.data[self.layout.linearize(&idx.idx)]
    }

    pub fn set(
        &mut self,
        idx: &CellIndex<2>,
        value: T
    ) {
        self.data[self.layout.linearize(&idx.idx)] = value;
    }
}

impl Field<f32, CellIndex<2>, CellLayout<2>>
{
    pub fn sample(
        &self,
        pos: [f32; 2],
        dx: [f32; 2],
    ) -> f32 {

        let gx = pos[0] / dx[0] - 0.5;
        let gy = pos[1] / dx[1] - 0.5;

        let i0 = gx.floor() as isize;
        let j0 = gy.floor() as isize;

        let tx = gx - i0 as f32;
        let ty = gy - j0 as f32;

        let i1 = i0 + 1;
        let j1 = j0 + 1;

        let v00 = self.sample_safe(
            i0,
            j0,
        );

        let v10 = self.sample_safe(
            i1,
            j0,
        );

        let v01 = self.sample_safe(
            i0,
            j1,
        );

        let v11 = self.sample_safe(
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
        i: isize,
        j: isize,
    ) -> &f32 {
        let dims = self.layout.dimensions();

        let ii = i.clamp(0, dims[0] as isize - 1) as usize;

        let jj = j.clamp(0, dims[1] as isize - 1) as usize;

        let cell =
            CellIndex::<2>::from_index(
                [ii, jj],
            );

        &self.data[self.layout.linearize(&cell.idx)]
    }
}

impl<T> Field<T, FaceIndex<2>, FaceLayout<2>> 
where
    T: Copy + Default
{
    pub fn get(
        &self,
        idx: &FaceIndex<2>
    ) -> T {
        let index = self.layout.linearize(&idx.idx);
        if  index >= self.data.len() {
            tracing::error!(": Index out of bounds in field. Length is {} but index is {}.", self.data.len(), index);
            return T::default();
        }
        self.data[index]
    }

    pub fn set(
        &mut self,
        idx: &FaceIndex<2>,
        value: T
    ) {
        self.data[self.layout.linearize(&idx.idx)] = value;
    }
}

impl Field<f32, FaceIndex<2>, FaceLayout<2>>
{
    pub fn sample(
        &self,
        pos: [f32; 2],
        dx: [f32; 2]
    ) -> f32 {
        let offset = if self.layout.get_axis() == 0 {
            [0.0, 0.5]
        } else {
            [0.5, 0.0]
        };

        let gx = pos[0] / dx[0] - offset[0];
        let gy = pos[1] / dx[1] - offset[1];

        let i0 = gx.floor() as isize;
        let j0 = gy.floor() as isize;

        let tx = gx - i0 as f32;
        let ty = gy - j0 as f32;

        let i1 = i0 + 1;
        let j1 = j0 + 1;

        let v00 = self.sample_safe(
            i0,
            j0,
        );

        let v10 = self.sample_safe(
            i1,
            j0,
        );

        let v01 = self.sample_safe(
            i0,
            j1,
        );

        let v11 = self.sample_safe(
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
        i: isize,
        j: isize,
    ) -> &f32 {

        let dims = self.layout.dimensions();

        let ii = i.clamp(0, dims[0] as isize - 1)
            as usize;

        let jj = j.clamp(0, dims[1] as isize - 1)
            as usize;

        let face =
            FaceIndex::<2>::new(
                self.layout.get_axis(),
                [ii, jj],
            );

        &self.data[self.layout.linearize(&face.idx)]
    }
}