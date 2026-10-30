use std::{collections::HashMap, sync::Arc};

use strum::IntoEnumIterator;
use wgpu::{BufferDescriptor, BufferUsages, CommandEncoder, CommandEncoderDescriptor};

use crate::{app::gpu_context::GPUContext, simulation::BackendDevice};

pub mod grid;

pub struct Frame<'a> {
    pub data: &'a mut SimulationData,
    pub command_encoder: &'a mut CommandEncoder,
    pub rendered_field: ScalarField,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, strum_macros::IntoStaticStr, strum_macros::EnumIter)]
pub enum ScalarField {
    Pressure,
    Divergence,
    Velocity
}

impl std::fmt::Display for ScalarField {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", <&str>::from(self))
    }
}

pub struct SimulationData {
    pub grid: grid::Grid2D,
    pub scalar_fields: HashMap<ScalarField, wgpu::Buffer>,

    pub velocity: [wgpu::Buffer; 2],

    pub is_solid: wgpu::Buffer,

    //off_diag: wgpu::Buffer,
    //diag: wgpu::Buffer,
}

impl SimulationData {
    pub fn new(gpu_context: &Arc<GPUContext>, grid_size: [usize; 2]) -> Self {
        let cell_count = grid_size[0] * grid_size[1];
        let x_face_count = (grid_size[0]+1) * grid_size[1];
        let y_face_count = grid_size[0] * (grid_size[1]+1);

        let cell_buffer_size = std::mem::size_of::<f32>() as u64 * cell_count as u64;

        let scalar_fields: HashMap<ScalarField, wgpu::Buffer> = ScalarField::iter()
            .map(|f| (f, 
                gpu_context.device.create_buffer(
                    &BufferDescriptor{
                        label: Some(&format!("{} buffer", <&str>::from(f))),
                        size: cell_buffer_size,
                        usage: BufferUsages::COPY_DST | BufferUsages::STORAGE | BufferUsages::COPY_SRC,
                        mapped_at_creation: false,
                    }
                )
            )).collect();

        
        let is_solid = gpu_context.device.create_buffer(
            &BufferDescriptor{
                label: Some("IsSolid Buffer"),
                size: std::mem::size_of::<u32>() as u64 * cell_count as u64,
                usage: BufferUsages::COPY_DST | BufferUsages::STORAGE,
                mapped_at_creation: false,
            }
        );

        let velocity_x = gpu_context.device.create_buffer(
            &BufferDescriptor{
                label: Some("X Velocity Buffer"),
                size: std::mem::size_of::<f32>() as u64 * x_face_count as u64,
                usage: BufferUsages::COPY_DST | BufferUsages::STORAGE | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }
        );
        let velocity_y = gpu_context.device.create_buffer(
            &BufferDescriptor{
                label: Some("Y Velocity Buffer"),
                size: std::mem::size_of::<f32>() as u64 * y_face_count as u64,
                usage: BufferUsages::COPY_DST | BufferUsages::STORAGE | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }
        );

        Self {
            grid: grid::Grid2D::new(grid_size, [1.0 / grid_size[0] as f32, 1.0 / grid_size[1] as f32]),
            scalar_fields,
            is_solid,
            velocity: [
                velocity_x,
                velocity_y
            ]
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
    pub grid_size: [usize; 2],
    pub dx: [f32; 2],
    pub is_running: bool,
    pub step: bool,
    pub solver_id: usize,
    pub solver_settings: SolverSettings,
    pub device: BackendDevice,
}

impl Default for SimulationSettings {
    fn default() -> Self {
        Self {
            grid_size: [50, 50],
            dx: [1.0/50.0, 1.0/50.0],
            solver_id: 1,
            step: Default::default(),
            is_running: false,
            solver_settings: Default::default(),
            device: BackendDevice::GPU,
        }
    }
}