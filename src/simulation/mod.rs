use crate::simulation::{data::SimulationData, pipeline::{SimulationPipeline, operator::{self, solver_registry::{SolverDescriptor, SolverRegistry}}}};
use colored::*;

pub mod data;
pub mod pipeline;

#[derive(Debug)]
pub struct SolverSettings {
    pub maxit: usize,
    pub tolerance: f32,
}

impl Default for SolverSettings {
    fn default() -> Self {
        Self {
            maxit: 1000,
            tolerance: 0.005
        }
    }
}

#[derive(Debug)]
pub struct SimulationSettings {
    pub is_running: bool,
    pub step: bool,
    pub solver_id: usize,
    pub solver_settings: SolverSettings,
}

impl Default for SimulationSettings {
    fn default() -> Self {
        Self {
            solver_id: 1,
            step: Default::default(),
            is_running: false,
            solver_settings: Default::default(),
        }
    }
}

pub struct PhysicalConstants {
    pub density: f32,
}

impl Default for PhysicalConstants {
    fn default() -> Self {
        Self { density: 1.0 }
    }
}

pub struct SimulationController {
    pub data: SimulationData,
    pub solver_registry: SolverRegistry,
    pipeline: SimulationPipeline,

    physical_constants: PhysicalConstants,
    current_solver: usize,
}

impl SimulationController {
    pub fn new() -> Self {
        let mut solver_registry = SolverRegistry::new();
        solver_registry.register(
            SolverDescriptor {
                id: 1,
                name: "Conjugate Gradient",
                create: operator::pressure_solution::cg_solver
            }
        );
        solver_registry.register(
            SolverDescriptor {
                id: 2,
                name: "Gauss Seidel",
                create: operator::pressure_solution::gs_solver
            }
        );

        let grid_size = [78, 56];
        let mut data = SimulationData::new(grid_size);
        operator::set_solid_cells_startup(&mut data);
        
        let mut pipeline = SimulationPipeline::new(
            operator::pressure_solution::cg_solver()
        );
        let solver_ref = pipeline.solver_ref();

        pipeline.add_step(operator::advect_velocities())
            .add_step(operator::apply_boundary_condition())
            .add_step(operator::apply_internal_condition())
            .add_step(operator::compute_divergence())
            .add_step(operator::pressure_solver(solver_ref))
            .add_step(operator::project_velocities())
            .add_step(operator::apply_internal_condition())
            .add_step(operator::compute_divergence())
            .add_step(operator::divergence_check());


        Self {
            data: data,
            pipeline,
            solver_registry,
            physical_constants: Default::default(),
            current_solver: 1,
        }
    }

    pub fn step(&mut self, settings: &SimulationSettings) {
        if settings.solver_id != self.current_solver {
            match self.solver_registry.create_solver(settings.solver_id) {
                Err(msg) => println!("{}", format!("ERROR: {}", msg).red()),
                Ok(new_solver) => {
                    match self.pipeline.set_solver(new_solver) {
                        Err(msg) => println!("{}", format!("ERROR: {}", msg).red()),
                        Ok(()) => {},
                    };
                    self.current_solver = settings.solver_id;
                    println!("{}", format!("Changed solver to {}", self.solver_registry.get_solver_name(settings.solver_id).unwrap()).blue());
                }
            }
        }
        if !settings.is_running && !settings.step {
            return;
        }
        match self.pipeline.modify_solver_settings(&settings.solver_settings) {
            Err(msg) => println!("{}", format!("ERROR: {}", msg).red()),
            Ok(()) => {},
        };
        

        let dt = 0.01f32;

        match self.pipeline.run(&mut self.data, &self.physical_constants, dt) {
            Err(msg) => println!("{}", format!("ERROR: Could not execute pipeline: {}", msg).red()),
            Ok(_time) => {},// println!("{}", format!("Pipeline succeeded after {} micros", time).green()),
        }
    }
}