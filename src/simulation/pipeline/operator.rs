use crate::simulation::data::{SimulationData, grid::Grid2D};

mod advection;
mod boundary_condition;
mod divergence_computation;
mod pressure_solution;
mod velocity_projection;

pub trait SimulationOperator {
    fn execute(
        &mut self,
        data: &mut SimulationData,
        context: &mut Grid2D,
    ) -> Result<u128, String>;

    fn get_name(&self) -> &'static str;
}

pub fn apply_boundary_condition() -> Box<dyn SimulationOperator> {
    return Box::new(boundary_condition::ApplyDomainBoundaryCondition::from_config("config/grid_topology.toml"));
}

pub fn apply_internal_condition() -> Box<dyn SimulationOperator> {
    return Box::new(boundary_condition::ApplyInteriorBoundaryCondition{});
}

pub fn advect_velocities() -> Box<dyn SimulationOperator> {
    return Box::new(advection::AdvectVelocity{});
}

pub fn compute_divergence() -> Box<dyn SimulationOperator> {
    return Box::new(divergence_computation::DivergenceComputation{});
}

pub fn solve_pressure() -> Box<dyn SimulationOperator> {
    return Box::new(pressure_solution::cpu_solver::CGSolver::new());
}

pub fn project_velocities() -> Box<dyn SimulationOperator> {
    return Box::new(velocity_projection::VelocityProjector{});
}