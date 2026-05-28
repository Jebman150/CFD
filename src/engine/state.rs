pub mod field;

use crate::engine::grid::multi_index::{CellIndex, FaceIndex};

use super::grid::Grid2D;

use field::{Field, CellType};

pub struct State2D {
    pub pressure: Field<f32, CellIndex<2>>,
    pub smoke: Field<f32, CellIndex<2>>,

    pub velocity: [Field<f32, FaceIndex<2>>; 2],

    pub cell_types: Field<CellType, CellIndex<2>>,
}

impl State2D {
    pub fn new(topology: &Grid2D) -> Self {
        Self {
            pressure: Field::<f32, CellIndex<2>>::new(topology.get_total_cell_count()),
            smoke: Field::<f32, CellIndex<2>>::new(topology.get_total_cell_count()),
            velocity: [
                Field::<f32, FaceIndex<2>>::new(topology.get_total_face_count(0)),
                Field::<f32, FaceIndex<2>>::new(topology.get_total_face_count(1))
            ],

            cell_types: Field::<CellType, CellIndex<2>>::new(topology.get_total_cell_count())
        }
    }
}