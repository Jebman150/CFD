use crate::simulation::{
    cpu_backend::{
        pipeline::operator::pressure_solution::SharedPressureSolver,
        data::CPUSimulationData
    },
    data::PhysicalConstants
};


mod advection;
mod boundary_condition;
mod divergence_computation;
pub mod pressure_solution;
mod velocity_projection;

pub mod solver_registry;

pub trait SimulationOperator {
    fn execute(
        &mut self,
        data: &mut CPUSimulationData,
        physical_constants: &PhysicalConstants,
        dt: f32,
    ) -> Result<u128, String>;

    fn get_name(&self) -> &'static str;
}

pub fn apply_boundary_condition() -> Box<dyn SimulationOperator> {
    return Box::new(boundary_condition::ApplyDomainBoundaryCondition::from_config("config/grid_topology.toml"));
}

pub fn apply_internal_condition() -> Box<dyn SimulationOperator> {
    return Box::new(boundary_condition::ApplyInteriorBoundaryCondition{});
}

pub fn set_solid_cells_startup(data: &mut CPUSimulationData) {
    boundary_condition::ApplyInteriorBoundaryCondition::set_solid_border(data);
    boundary_condition::ApplyInteriorBoundaryCondition::set_solid_obstacle(data);
    boundary_condition::ApplyInteriorBoundaryCondition::precompute_edge_flow_from_solid_cells(data);
}

pub fn advect_velocities() -> Box<dyn SimulationOperator> {
    return Box::new(advection::AdvectVelocity{});
}

pub fn compute_divergence() -> Box<dyn SimulationOperator> {
    return Box::new(divergence_computation::DivergenceComputation{});
}

pub fn divergence_check() -> Box<dyn SimulationOperator> {
    return Box::new(divergence_computation::DivergenceChecker{});
}

pub fn velocity_abs_val_calculation() -> Box<dyn SimulationOperator> {
    return Box::new(velocity_projection::VelocityAbsValCalculation{});
}

pub fn pressure_solver(selected_solver: SharedPressureSolver) -> Box<dyn SimulationOperator> {
    return Box::new(pressure_solution::SolverStep::new(selected_solver));
}

pub fn project_velocities() -> Box<dyn SimulationOperator> {
    return Box::new(velocity_projection::VelocityProjector{});
}