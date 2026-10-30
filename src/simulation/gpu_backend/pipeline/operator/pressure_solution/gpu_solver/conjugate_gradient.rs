use anyhow::Error;

use crate::simulation::{data::{ScalarField, SimulationData}, gpu_backend::pipeline::gpu_helper::{self, ComputeKernel}};

pub struct CGVars {
    alpha: f32,
    beta: f32,
    rs: f32,
    pad: f32
}

pub fn construct_divergence_preprocessing_kernel(
    device: &std::sync::Arc<wgpu::Device>,
    data: &SimulationData,
    parameter_buffer: &wgpu::Buffer,
    r: &wgpu::Buffer,
    p: &wgpu::Buffer,
    reduction_input: &wgpu::Buffer
) -> anyhow::Result<ComputeKernel> {
    let mut layout_buffer_entries = Vec::new();
    gpu_helper::add_layout_buffer(&mut layout_buffer_entries, 0, false);
    gpu_helper::add_layout_uniform(&mut layout_buffer_entries, 1);
    gpu_helper::add_layout_buffer(&mut layout_buffer_entries, 2, false);
    gpu_helper::add_layout_buffer(&mut layout_buffer_entries,4, false);
    gpu_helper::add_layout_buffer(&mut layout_buffer_entries, 6, false);
    gpu_helper::add_layout_buffer(&mut layout_buffer_entries, 7, false);

    let cell_num = data.grid.get_dimensions()[0] * data.grid.get_dimensions()[1];

    ComputeKernel::new("Divergence preprocessor")
        .build_pipeline(
            device,
            "shaders/compute/pressure_preprocessing.wgsl",
            "process_divergence",
            &layout_buffer_entries,
            &vec![
                (0, data.scalar_fields.get(&ScalarField::Divergence).ok_or(Error::msg("No divergence field initialized!"))?),
                (1, parameter_buffer),
                (2, data.scalar_fields.get(&ScalarField::Pressure).ok_or(Error::msg("No pressure field initialized!"))?),
                (4, r),
                (6, reduction_input),
                (7, p)
            ],
            cell_num,
        )
}

pub fn construct_cg_vars_initializer_kernel(
    device: &std::sync::Arc<wgpu::Device>,
    _data: &SimulationData,
    reduction_output: &wgpu::Buffer,
    cg_vars: &wgpu::Buffer
) -> anyhow::Result<ComputeKernel> {
    let mut layout_buffer_entries = Vec::new();
    gpu_helper::add_layout_buffer(&mut layout_buffer_entries, 3, true);
    gpu_helper::add_layout_buffer(&mut layout_buffer_entries,5, false);

    ComputeKernel::new("CG vars initializer")
        .build_pipeline(
            device,
            "shaders/compute/pressure_preprocessing.wgsl",
            "set_initial_cg_vars",
            &layout_buffer_entries,
            &vec![
                (3, reduction_output),
                (5, cg_vars)
            ],
            1,
        )
}

pub fn construct_apply_kernel(
    device: &std::sync::Arc<wgpu::Device>,
    data: &SimulationData,
    parameter_buffer: &wgpu::Buffer,
    p: &wgpu::Buffer,
    ap: &wgpu::Buffer,
    reduction_input: &wgpu::Buffer,
) -> anyhow::Result<ComputeKernel> {
    let mut layout_buffer_entries = Vec::new();
    gpu_helper::add_layout_uniform(&mut layout_buffer_entries, 2);
    gpu_helper::add_layout_buffer(&mut layout_buffer_entries, 3, true);
    gpu_helper::add_layout_buffer(&mut layout_buffer_entries, 5, false);
    gpu_helper::add_layout_buffer(&mut layout_buffer_entries, 10, false);
    gpu_helper::add_layout_buffer(&mut layout_buffer_entries, 11, false);

    let cell_num = data.grid.get_dimensions()[0] * data.grid.get_dimensions()[1];

    ComputeKernel::new("Laplacian apply")
        .build_pipeline(
            device,
            "shaders/compute/pressure_solver.wgsl",
            "apply",
            &layout_buffer_entries,
            &vec![
                (2, parameter_buffer),
                (3, &data.is_solid),
                (5, ap),
                (10, p),
                (11, reduction_input)
            ],
            cell_num,
        )
}

pub fn construct_reduction_kernel(
    device: &std::sync::Arc<wgpu::Device>,
    data: &SimulationData,
    reduction_input: &wgpu::Buffer,
    reduction_output: &wgpu::Buffer,
) -> anyhow::Result<ComputeKernel> {
    let mut layout_buffer_entries = Vec::new();
    gpu_helper::add_layout_buffer(&mut layout_buffer_entries, 0, true);
    gpu_helper::add_layout_buffer(&mut layout_buffer_entries, 1, false);

    let cell_num = data.grid.get_dimensions()[0] * data.grid.get_dimensions()[1];

    ComputeKernel::new("Reduction kernel")
        .build_pipeline(
            device,
            "shaders/compute/partial_reduction.wgsl",
            "main",
            &layout_buffer_entries,
            &vec![
                (0, reduction_input),
                (1, reduction_output),
            ],
            cell_num,
        )
}

pub fn construct_update_xr_kernel(
    device: &std::sync::Arc<wgpu::Device>,
    data: &SimulationData,
    ap_product: &wgpu::Buffer,
    reduction_output: &wgpu::Buffer,
    p: &wgpu::Buffer,
    r: &wgpu::Buffer,
    cg_vars: &wgpu::Buffer,
    reduction_input: &wgpu::Buffer,
) -> anyhow::Result<ComputeKernel> {
    let mut layout_buffer_entries = Vec::new();
    gpu_helper::add_layout_buffer(&mut layout_buffer_entries, 0, false);
    gpu_helper::add_layout_buffer(&mut layout_buffer_entries, 5, false);
    gpu_helper::add_layout_buffer(&mut layout_buffer_entries, 6, false);
    gpu_helper::add_layout_buffer(&mut layout_buffer_entries, 8, false);
    gpu_helper::add_layout_buffer(&mut layout_buffer_entries, 9, false);
    gpu_helper::add_layout_buffer(&mut layout_buffer_entries, 10, false);
    gpu_helper::add_layout_buffer(&mut layout_buffer_entries, 11, false);

    let cell_num = data.grid.get_dimensions()[0] * data.grid.get_dimensions()[1];

    ComputeKernel::new("Update xr kernel")
        .build_pipeline(
            device,
            "shaders/compute/pressure_solver.wgsl",
            "update_xr",
            &layout_buffer_entries,
            &vec![
                (0,data.scalar_fields.get(&ScalarField::Pressure).ok_or(Error::msg("No pressure field initialized!"))?),
                (5, ap_product),
                (6, r),
                (8, cg_vars),
                (9, reduction_output),
                (10, p),
                (11, reduction_input)
            ],
            cell_num,
        )
}

pub fn construct_update_p_kernel(
    device: &std::sync::Arc<wgpu::Device>,
    data: &SimulationData,
    reduction_output: &wgpu::Buffer,
    p: &wgpu::Buffer,
    r: &wgpu::Buffer,
    cg_vars: &wgpu::Buffer,
) -> anyhow::Result<ComputeKernel> {
    let mut layout_buffer_entries = Vec::new();
    gpu_helper::add_layout_buffer(&mut layout_buffer_entries, 6, false);
    gpu_helper::add_layout_buffer(&mut layout_buffer_entries, 7, true);
    gpu_helper::add_layout_buffer(&mut layout_buffer_entries, 8, false);
    gpu_helper::add_layout_buffer(&mut layout_buffer_entries, 10, false);

    let cell_num = data.grid.get_dimensions()[0] * data.grid.get_dimensions()[1];

    ComputeKernel::new("Update p kernel")
        .build_pipeline(
            device,
            "shaders/compute/pressure_solver.wgsl",
            "update_p",
            &layout_buffer_entries,
            &vec![
                (6, r),
                (7, reduction_output),
                (8, cg_vars),
                (10, p)
            ],
            cell_num,
        )
}