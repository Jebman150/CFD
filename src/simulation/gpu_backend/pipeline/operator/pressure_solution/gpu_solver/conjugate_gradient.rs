use anyhow::Error;

use crate::simulation::{data::{ScalarField, SimulationData}, gpu_backend::pipeline::gpu_helper::{self, ComputeKernel}};


pub fn construct_apply_kernel(
    device: &std::sync::Arc<wgpu::Device>,
    data: &SimulationData,
    parameter_buffer: &wgpu::Buffer,
    diagonal_entries: &wgpu::Buffer,
    flow_allowed_bit_mask: &wgpu::Buffer,
    ap: &wgpu::Buffer,
) -> anyhow::Result<ComputeKernel> {
    let mut layout_buffer_entries = Vec::new();
    gpu_helper::add_layout_buffer(&mut layout_buffer_entries, 0, true);
    gpu_helper::add_layout_buffer(&mut layout_buffer_entries, 1, true);
    gpu_helper::add_layout_uniform(&mut layout_buffer_entries, 2);
    gpu_helper::add_layout_buffer(&mut layout_buffer_entries, 3, true);
    gpu_helper::add_layout_buffer(&mut layout_buffer_entries, 4, true);
    gpu_helper::add_layout_buffer(&mut layout_buffer_entries, 5, false);

    let cell_num = data.grid.get_dimensions()[0] * data.grid.get_dimensions()[1];

    ComputeKernel::new("Laplacian apply")
        .build_pipeline(
            device,
            "shaders/compute/pressure_solver.wgsl",
            "apply",
            &layout_buffer_entries,
            &vec![
                (0, data.scalar_fields.get(&ScalarField::Pressure).ok_or(Error::msg("No divergence field initialized!"))?),
                (1, data.scalar_fields.get(&ScalarField::Divergence).ok_or(Error::msg("No divergence field initialized!"))?),
                (2, parameter_buffer),
                (3, diagonal_entries),
                (4, flow_allowed_bit_mask),
                (5, ap),
            ],
            cell_num,
        )
}