use crate::simulation::data::SimulationData;

pub mod cpu_solver;

pub trait Solver {
    fn compute(
        &mut self,
        data: &mut SimulationData,
        dt: f32,
    ) -> usize;

    fn set_tolerance(&mut self, val: f32);
    fn set_maxit(&mut self, val: usize);
}

pub fn create_cg_solver() -> Box<dyn Solver> {
    Box::new(cpu_solver::CGSolver::new())
}