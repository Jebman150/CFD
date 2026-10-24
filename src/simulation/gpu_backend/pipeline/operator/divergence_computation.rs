use anyhow::Error;

use crate::simulation::{data::{ScalarField, SimulationData}, gpu_backend::pipeline::{gpu_helper::{self, ComputeKernel}, operator::SimulationOperator}};

pub fn construct_divergence_calculator_kernel(
    device: &std::sync::Arc<wgpu::Device>,
    data: &SimulationData,
    parameter_buffer: &wgpu::Buffer,
) -> anyhow::Result<Box<dyn SimulationOperator>> {
    let mut layout_buffer_entries = Vec::new();
    gpu_helper::add_layout_buffer(&mut layout_buffer_entries, 0, true);
    gpu_helper::add_layout_buffer(&mut layout_buffer_entries, 1, true);
    gpu_helper::add_layout_buffer(&mut layout_buffer_entries, 2, false);
    gpu_helper::add_layout_uniform(&mut layout_buffer_entries, 3);

    let cell_num = data.grid.get_dimensions()[0] * data.grid.get_dimensions()[1];

    Ok(Box::new(ComputeKernel::new("Divergence computation")
        .build_pipeline(
            device,
            "shaders/compute/divergence_computation.wgsl",
            "compute_divergence",
            &layout_buffer_entries,
            &vec![
                (0, &data.velocity[0]),
                (1, &data.velocity[1]),
                (2, data.scalar_fields.get(&ScalarField::Divergence).ok_or(Error::msg("No divergence field initialized!"))?),
                (3, parameter_buffer)
            ],
            cell_num,
        )?))
}