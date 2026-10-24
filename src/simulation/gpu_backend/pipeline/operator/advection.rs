use crate::simulation::{data::SimulationData, gpu_backend::pipeline::{gpu_helper::{self, ComputeKernel}, operator::SimulationOperator}};


pub fn construct_velocity_avection_kernel(
    device: &std::sync::Arc<wgpu::Device>,
    data: &SimulationData,
    velocity_temp_buffer: &[wgpu::Buffer; 2],
    parameter_buffer: &wgpu::Buffer,
) -> anyhow::Result<Box<dyn SimulationOperator>> {
    let mut layout_buffer_entries = Vec::new();
    gpu_helper::add_layout_buffer(&mut layout_buffer_entries, 0, true);
    gpu_helper::add_layout_buffer(&mut layout_buffer_entries, 1, true);
    gpu_helper::add_layout_buffer(&mut layout_buffer_entries, 2, false);
    gpu_helper::add_layout_buffer(&mut layout_buffer_entries, 3, false);
    gpu_helper::add_layout_uniform(&mut layout_buffer_entries, 4);

    let x_face_num = (data.grid.get_dimensions()[0]+1) * data.grid.get_dimensions()[1];
    let y_face_num = data.grid.get_dimensions()[0] * (data.grid.get_dimensions()[1]+1);

    Ok(Box::new(ComputeKernel::new("Velocity advection")
        .build_pipeline(
            device,
            "shaders/compute/velocity_advection.wgsl",
            "velocity_advection",
            &layout_buffer_entries,
            &vec![
                (0, &velocity_temp_buffer[0]),
                (1, &velocity_temp_buffer[1]),
                (2, &data.velocity[0]),
                (3, &data.velocity[1]),
                (4, parameter_buffer)
            ],
            x_face_num + y_face_num,
        )?))
}