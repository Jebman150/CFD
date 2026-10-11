use crate::simulation::{data::{SimulationData, grid::Grid2D}, pipeline::SimulationPipeline, solver_registry::{SolverDescriptor, SolverRegistry}};
use colored::*;

pub mod solver;
pub mod data;
pub mod solver_registry;
pub mod pipeline;

#[derive(Debug, Default)]
pub struct SimulationSettings {
    pub is_running: bool,
    pub step: bool,
}

pub struct SimulationController {
    pub topology: Grid2D,
    pub data: SimulationData,
    solver_registry: SolverRegistry,
    pipeline: SimulationPipeline,
}

impl SimulationController {
    pub fn new() -> Self {
        let mut solver_registry = SolverRegistry::new();
        solver_registry.register(
            SolverDescriptor {
                id: 1,
                name: "Conjugate Gradient",
                create: solver::create_cg_solver
            }
        );

        let mut pipeline = SimulationPipeline::new();
        pipeline.add_step(pipeline::operator::advect_velocities())
            //.add_step(pipeline::operator::apply_boundary_condition())
            .add_step(pipeline::operator::apply_internal_condition())
            .add_step(pipeline::operator::compute_divergence())
            .add_step(pipeline::operator::solve_pressure())
            .add_step(pipeline::operator::project_velocities())
            .add_step(pipeline::operator::apply_internal_condition());

        let topology = data::grid::Grid2D::new([100, 100], [0.01, 0.01]);

        Self {
            data: SimulationData::new(&topology),
            topology,
            pipeline,
            solver_registry
        }
    }

    pub fn step(&mut self, settings: &SimulationSettings) {
        if !settings.is_running && !settings.step {
            return;
        }

        match self.pipeline.run(&mut self.data, &mut self.topology) {
            Err(msg) => println!("{}", format!("ERROR: Could not execute pipeline: {}", msg).red()),
            Ok(time) => println!("Pipeline succeeded after {} micros", time),
        }
    }
}