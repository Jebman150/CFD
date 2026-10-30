use crate::simulation::{data::SimulationData, gpu_backend::pipeline::{gpu_helper::{self, ComputeKernel}, operator::SimulationOperator}};



pub fn construct_boundary_condition_kernel(
    device: &std::sync::Arc<wgpu::Device>,
    data: &SimulationData,
    parameter_buffer: &wgpu::Buffer,
) -> anyhow::Result<Box<dyn SimulationOperator>> {
    let mut layout_buffer_entries = Vec::new();
    gpu_helper::add_layout_buffer(&mut layout_buffer_entries, 0, true);
    gpu_helper::add_layout_buffer(&mut layout_buffer_entries, 1, false);
    gpu_helper::add_layout_buffer(&mut layout_buffer_entries, 2, false);
    gpu_helper::add_layout_uniform(&mut layout_buffer_entries, 3);

    let cell_num = data.grid.get_dimensions()[0] * data.grid.get_dimensions()[1];

    Ok(Box::new(ComputeKernel::new("Boundary condition apply")
        .build_pipeline(
            device,
            "shaders/compute/apply_boundary_condition.wgsl",
            "apply_solid_cells",
            &layout_buffer_entries,
            &vec![
                (0, &data.is_solid),
                (1, &data.velocity[0]),
                (2, &data.velocity[1]),
                (3, parameter_buffer)
            ],
            cell_num
        )?))
}