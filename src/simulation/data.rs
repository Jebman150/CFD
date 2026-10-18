use crate::simulation::data::{field::{CellField2, CellField2F, FaceField2F}, grid::{Grid, Grid2D, layout::{CellLayout, FaceLayout}}};

pub mod field;
pub mod grid;

pub struct SimulationData {
    pub grid: Grid<2>,
    
    pub pressure: CellField2F,
    pub divergence: CellField2F,
    pub smoke: CellField2F,

    pub velocity: [FaceField2F; 2],

    pub is_solid: CellField2<bool>,
    pub flow_edge_allowed: CellField2<i8>,
    pub flow_edge_count: [CellField2<i8>; 2],
}

impl SimulationData {
    pub fn new(dim: [usize; 2]) -> Self {
        let cell_layout = CellLayout::<2>::new(dim);
        let x_face_layout = FaceLayout::<2>::new(dim, 0);
        let y_face_layout = FaceLayout::<2>::new(dim, 1);


        Self {
            grid: Grid2D::new(dim, [1.0 / dim[0] as f32, 1.0 / dim[1] as f32]),
            pressure: CellField2F::new(&cell_layout),
            divergence: CellField2F::new(&cell_layout),
            smoke: CellField2F::new(&cell_layout),
            velocity: [
                FaceField2F::new(&x_face_layout),
                FaceField2F::new(&y_face_layout)
            ],

            is_solid: CellField2::<bool>::new(&cell_layout),
            flow_edge_allowed: CellField2::<i8>::new(&cell_layout),
            flow_edge_count: [
                CellField2::<i8>::new(&cell_layout),
                CellField2::<i8>::new(&cell_layout)
            ],
        }
    }
}