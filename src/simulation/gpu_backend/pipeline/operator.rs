use wgpu::CommandEncoder;


pub mod advection;
pub mod boundary_condition;
pub mod divergence_computation;
pub mod pressure_solution;
pub mod velocity_projection;

pub mod solver_registry;

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
pub struct Parameter {
    pub grid_size: [u32; 2],
    pub dx: [f32; 2],
    pub density: f32,
    pub dt: f32,
}

pub trait SimulationOperator {

    fn encode(
        &self,
        encoder: &mut CommandEncoder
    ) -> anyhow::Result<()>;
    

    fn get_name(&self) -> &'static str;
}