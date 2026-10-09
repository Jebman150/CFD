use crate::simulation::data::{field::{CellType, Field}, grid::{Grid, Grid2D, multi_index::{CellIndex, FaceIndex}}};

pub mod field;
pub mod grid;

pub struct SimulationData {
    pub topology: Grid<2>,
    
    pub pressure: Field<f32, CellIndex<2>>,
    pub divergence: Field<f32, CellIndex<2>>,
    pub smoke: Field<f32, CellIndex<2>>,

    pub velocity: [Field<f32, FaceIndex<2>>; 2],

    pub cell_types: Field<CellType, CellIndex<2>>,
}

impl SimulationData {
    pub fn new(topology: &Grid<2>) -> Self {
        Self {
            topology: topology.clone(),
            pressure: Field::<f32, CellIndex<2>>::new(topology.get_total_cell_count()),
            divergence: Field::<f32, CellIndex<2>>::new(topology.get_total_cell_count()),
            smoke: Field::<f32, CellIndex<2>>::new(topology.get_total_cell_count()),
            velocity: [
                Field::<f32, FaceIndex<2>>::new(topology.get_total_face_count(0)),
                Field::<f32, FaceIndex<2>>::new(topology.get_total_face_count(1))
            ],

            cell_types: Field::<CellType, CellIndex<2>>::new(topology.get_total_cell_count())
        }
    }

    pub fn print_velocity(&self, context: &Grid2D, axis: usize) {
        let mut current_y = 0;
        print!("{}", current_y);
        for mut face_idx in context.face_iter(axis) {
            face_idx.axis = axis;
            if face_idx.idx[1] != current_y {
                current_y = face_idx.idx[1];
                println!("");
                print!("{}", current_y);
            }
            print!("{} ", self.velocity[axis].get(context, &face_idx));
        }
    }
}