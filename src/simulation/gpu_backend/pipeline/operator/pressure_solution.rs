use std::collections::HashMap;

use wgpu::BufferUsages;

use crate::simulation::{data::SimulationData, gpu_backend::pipeline::operator::SimulationOperator};

mod gpu_solver;

pub fn construct_cg_solver_kernel(
    device: &std::sync::Arc<wgpu::Device>,
    data: &SimulationData,
    parameter_buffer: &wgpu::Buffer,
) -> anyhow::Result<Box<dyn SimulationOperator>> {
    let cell_num = data.grid.get_dimensions()[0] * data.grid.get_dimensions()[1];

    let diag_entries = device.create_buffer(&wgpu::BufferDescriptor{
        label: Some("Diagonal laplacian entries"),
        size: (std::mem::size_of::<f32>() * cell_num) as u64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let off_diag_flow_bitmask = device.create_buffer(&wgpu::BufferDescriptor{
        label: Some("Offdiagonal laplacian flow bitmask"),
        size: (std::mem::size_of::<u32>() * cell_num) as u64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let ap_product = device.create_buffer(&wgpu::BufferDescriptor{
        label: Some("Ap product"),
        size: (std::mem::size_of::<f32>() * cell_num) as u64,
        usage: BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let apply_kernel = gpu_solver::conjugate_gradient::construct_apply_kernel(
        device, data, parameter_buffer,
        &diag_entries, &off_diag_flow_bitmask, &ap_product
    );

    
}

