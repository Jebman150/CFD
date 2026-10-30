use std::collections::HashMap;

use wgpu::BufferUsages;

use crate::simulation::{data::SimulationData, gpu_backend::pipeline::{gpu_helper::MultiStageKernel, operator::SimulationOperator}};

mod gpu_solver;

pub fn construct_cg_solver_kernel(
    device: &std::sync::Arc<wgpu::Device>,
    data: &SimulationData,
    parameter_buffer: &wgpu::Buffer,
) -> anyhow::Result<Box<dyn SimulationOperator>> {
    let cell_num = data.grid.get_dimensions()[0] * data.grid.get_dimensions()[1];

    let ap_product = device.create_buffer(&wgpu::BufferDescriptor{
        label: Some("Ap product"),
        size: (std::mem::size_of::<f32>() * cell_num) as u64,
        usage: BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let partial1 = device.create_buffer(&wgpu::BufferDescriptor{
        label: Some("Partial1 buffer"),
        size: (std::mem::size_of::<f32>() * (cell_num+1)) as u64,
        usage: BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let partial2 = device.create_buffer(&wgpu::BufferDescriptor{
        label: Some("Partial2 buffer"),
        size: (std::mem::size_of::<f32>() * (cell_num+1)) as u64,
        usage: BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let p = device.create_buffer(&wgpu::BufferDescriptor{
        label: Some("p helper buffer"),
        size: (std::mem::size_of::<f32>() * cell_num) as u64,
        usage: BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let r = device.create_buffer(&wgpu::BufferDescriptor{
        label: Some("Residual buffer"),
        size: (std::mem::size_of::<f32>() * cell_num) as u64,
        usage: BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let cg_vars = device.create_buffer(&wgpu::BufferDescriptor{
        label: Some("CG Vars"),
        size: std::mem::size_of::<gpu_solver::conjugate_gradient::CGVars>() as u64,
        usage: BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let reduction_iterations = (cell_num as f32).log(64.0).ceil() as usize;
    let reduction_output = if reduction_iterations % 2 == 0 {
        &partial1
    } else {
        &partial2
    };

    let reduction_1_to_2_kernel = gpu_solver::conjugate_gradient::construct_reduction_kernel(
        device, data, &partial1, &partial2
    )?;
    let reduction_2_to_1_kernel = gpu_solver::conjugate_gradient::construct_reduction_kernel(
        device, data, &partial2, &partial1
    )?;

    let divergence_preprocessor_kernel = gpu_solver::conjugate_gradient::construct_divergence_preprocessing_kernel(
        device, data, parameter_buffer, &r, &partial1, &p
    )?;

    let cg_vars_initializer_kernel = gpu_solver::conjugate_gradient::construct_cg_vars_initializer_kernel(
        device, data, reduction_output, &cg_vars
    )?;

    let mut preprocessor = vec![
        divergence_preprocessor_kernel
    ];
    let mut forward = true;
    for _ in 0..reduction_iterations {
        if forward {
            preprocessor.push(reduction_1_to_2_kernel.clone());
            forward = false;
        } else {
            preprocessor.push(reduction_2_to_1_kernel.clone());
            forward = true;
        }
    }
    preprocessor.push(cg_vars_initializer_kernel);

    let apply_kernel = gpu_solver::conjugate_gradient::construct_apply_kernel(
        device, data, parameter_buffer, &p, &ap_product, &partial1
    )?;

    let update_xr_kernel = gpu_solver::conjugate_gradient::construct_update_xr_kernel(
        device, data, &ap_product, reduction_output, &p, &r, &cg_vars, &partial1
    )?;

    let update_p_kernel = gpu_solver::conjugate_gradient::construct_update_p_kernel(
        device, data, reduction_output, &p, &r, &cg_vars
    )?;

    let mut stages = Vec::new();
    stages.push(apply_kernel);
    let mut forward = true;
    for _ in 0..reduction_iterations {
        if forward {
            stages.push(reduction_1_to_2_kernel.clone());
            forward = false;
        } else {
            stages.push(reduction_2_to_1_kernel.clone());
            forward = true;
        }
    }
    stages.push(update_xr_kernel);
    forward = true;
    for _ in 0..reduction_iterations {
        if forward {
            stages.push(reduction_1_to_2_kernel.clone());
            forward = false;
        } else {
            stages.push(reduction_2_to_1_kernel.clone());
            forward = true;
        }
    }
    stages.push(update_p_kernel);

    let mut buffer_map = HashMap::new();
    buffer_map.insert(0, ap_product);
    buffer_map.insert(1, partial1);
    buffer_map.insert(2, partial2);
    buffer_map.insert(3, p);
    buffer_map.insert(4, r);
    buffer_map.insert(5, cg_vars);

    Ok(Box::new(MultiStageKernel::new(
        "Conjugate Gradient Kernel",
        preprocessor,
        stages,
        vec![],
        100,
        buffer_map
    )))
}

