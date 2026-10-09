use crate::simulation::{data::SimulationData, pipeline::operator::SimulationOperator};

pub mod cpu_solver;

pub trait PressureSolver: SimulationOperator {
    fn compute(
        &mut self,
        data: &mut SimulationData,
        dt: f32,
    ) -> Result<usize, String>;

    fn set_tolerance(&mut self, val: f32);
    fn set_maxit(&mut self, val: usize);
}