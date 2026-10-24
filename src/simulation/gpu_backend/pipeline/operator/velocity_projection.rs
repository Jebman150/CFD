use anyhow::Error;

use crate::simulation::{data::{ScalarField, SimulationData}, gpu_backend::pipeline::{gpu_helper::{self, ComputeKernel}, operator::SimulationOperator}};

pub fn construct_velocity_projection_kernel(
    device: &std::sync::Arc<wgpu::Device>,
    data: &SimulationData,
    parameter_buffer: &wgpu::Buffer,
) -> anyhow::Result<Box<dyn SimulationOperator>> {
    let mut layout_buffer_entries = Vec::new();
    gpu_helper::add_layout_buffer(&mut layout_buffer_entries, 0, true);
    gpu_helper::add_layout_buffer(&mut layout_buffer_entries, 1, false);
    gpu_helper::add_layout_buffer(&mut layout_buffer_entries, 2, false);
    gpu_helper::add_layout_uniform(&mut layout_buffer_entries, 3);

    let x_face_num = (data.grid.get_dimensions()[0]+1) * data.grid.get_dimensions()[1];
    let y_face_num = data.grid.get_dimensions()[0] * (data.grid.get_dimensions()[1]+1);

    Ok(Box::new(ComputeKernel::new("Velocity projection")
        .build_pipeline(
            device,
            "shaders/compute/velocity_projection.wgsl",
            "velocity_projection",
            &layout_buffer_entries,
            &vec![
                (0, data.scalar_fields.get(&ScalarField::Pressure).ok_or(Error::msg("No pressure field initialized!"))?),
                (1, &data.velocity[0]),
                (2, &data.velocity[1]),
                (3, parameter_buffer)
            ],
            x_face_num + y_face_num,
        )?))
}

pub fn construct_velocity_abs_val_kernel(
    device: &std::sync::Arc<wgpu::Device>,
    data: &SimulationData,
    parameter_buffer: &wgpu::Buffer,
) -> anyhow::Result<Box<dyn SimulationOperator>> {
    let mut layout_buffer_entries = Vec::new();
    gpu_helper::add_layout_buffer(&mut layout_buffer_entries, 1, false);
    gpu_helper::add_layout_buffer(&mut layout_buffer_entries, 2, false);
    gpu_helper::add_layout_uniform(&mut layout_buffer_entries, 3);
    gpu_helper::add_layout_buffer(&mut layout_buffer_entries, 4, false);

    let x_face_num = (data.grid.get_dimensions()[0]+1) * data.grid.get_dimensions()[1];
    let y_face_num = data.grid.get_dimensions()[0] * (data.grid.get_dimensions()[1]+1);

    Ok(Box::new(ComputeKernel::new("Velocity absolute value calculation")
        .build_pipeline(
            device,
            "shaders/compute/velocity_projection.wgsl",
            "abs_velocity_value_calculation",
            &layout_buffer_entries,
            &vec![
                (1, &data.velocity[0]),
                (2, &data.velocity[1]),
                (3, parameter_buffer),
                (4, data.scalar_fields.get(&ScalarField::Velocity).ok_or(Error::msg("No velocity field initialized!"))?)
            ],
            x_face_num + y_face_num,
        )?))
}