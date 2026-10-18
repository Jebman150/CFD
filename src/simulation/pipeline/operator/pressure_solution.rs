use std::{sync::{Arc, Mutex}, time::Instant};

use crate::{logging::start_log_entry, simulation::{PhysicalConstants, data::SimulationData, pipeline::operator::SimulationOperator}};

mod cpu_solver;
mod linear_operator;

pub trait PressureSolver: std::fmt::Debug {
    fn compute(
        &self,
        data: &mut SimulationData,
        physical_constants: &PhysicalConstants,
        dt: f32,
    ) -> Result<usize, String>;

    fn set_tolerance(&mut self, val: f32);
    fn set_maxit(&mut self, val: usize);
}

pub type SharedPressureSolver = Arc<Mutex<Box<dyn PressureSolver>>>;

pub struct SolverStep {
    selected_solver: SharedPressureSolver,
}

impl SolverStep {
    pub fn new(selected_solver: Arc<Mutex<Box<dyn PressureSolver>>>) -> Self {
        Self {
            selected_solver
        }
    }
}

impl SimulationOperator for SolverStep {
    fn execute(
        &mut self,
        data: &mut SimulationData,
        physical_constants: &PhysicalConstants,
        dt: f32,
    ) -> Result<u128, String> {
        match start_log_entry("logs/solver/solver_error.log") {
            Ok(()) => {},
            Err(err) => return Err(format!("Could not start log line: {}", err)),
        };
        let start = Instant::now();
        puffin::profile_scope!("solve_pressure");
        self.selected_solver
            .lock()
            .map_err(|e| format!("Solver mutex poisoned: {}", e))?
            .compute(data, physical_constants, dt)?;
        Ok(start.elapsed().as_micros())
    }

    fn get_name(&self) -> &'static str {
        "Pressure solver"
    }
}

pub fn cg_solver() -> Box<dyn PressureSolver> {
    return Box::new(cpu_solver::conjugate_gradient::CGSolver::new());
}

pub fn gs_solver() -> Box<dyn PressureSolver> {
    return Box::new(cpu_solver::gauss_seidel::GSSolver::new());
}