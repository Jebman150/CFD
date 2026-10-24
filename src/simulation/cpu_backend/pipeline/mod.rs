use std::{sync::{Arc, Mutex}, time::Instant};

use operator::pressure_solution::{
    SharedPressureSolver,
    PressureSolver
};

use crate::simulation::{cpu_backend::data::CPUSimulationData, data::{
    PhysicalConstants,
    SolverSettings
}};

pub mod operator;

pub struct SimulationPipeline {
    operations: Vec<Box<dyn operator::SimulationOperator>>,
    pub selected_solver: SharedPressureSolver,
}

impl SimulationPipeline {
    pub fn new(initial_solver: Box<dyn PressureSolver>) -> Self {
        Self {
            operations: Vec::new(),
            selected_solver: Arc::new(Mutex::new(initial_solver))
        }
    }

    pub fn add_step(&mut self, op: Box<dyn operator::SimulationOperator>) -> &mut Self {
        self.operations.push(op);
        self
    }

    pub fn run(&mut self, data: &mut CPUSimulationData , physical_constants: &PhysicalConstants, dt: f32) -> Result<u128, String> {
        let start = Instant::now();
        for op in &mut self.operations {
            match op.execute(data, physical_constants, dt) {
                Ok(_time) => {},
                Err(msg) => {
                    tracing::error!("{} failed with msg {}", op.get_name(), msg);
                },
            }
        }
        Ok(start.elapsed().as_micros())
    }

    pub fn set_solver(&mut self, new_solver: Box<dyn PressureSolver>) -> Result<(), String> {
        let mut solver = self.selected_solver
            .lock()
            .map_err(|e| format!("Solver mutex poisoned: {}", e))?;

        *solver = new_solver;

        Ok(())
    }

    pub fn modify_solver_settings(&mut self, settings: &SolverSettings) -> Result<(), String> {
        let mut solver = self.selected_solver
            .lock()
            .map_err(|e| format!("Solver mutex poisoned: {}", e))?;

        solver.set_maxit(settings.maxit);
        solver.set_tolerance(settings.tolerance);

        Ok(())
    }

    pub fn solver_ref(&self) -> SharedPressureSolver {
        Arc::clone(&self.selected_solver)
    }
}