use std::{collections::HashMap, sync::Arc};


use wgpu::BufferUsages;

use crate::{app::gpu_context::GPUContext, simulation::{data::{Frame, PhysicalConstants, SimulationData}, gpu_backend::pipeline::{SimulationPipeline, operator::{self, Parameter, solver_registry::{SolverDescriptor, SolverRegistry}}}}};

mod pipeline;

pub struct GPUSimulationEngine {
    gpu_context: Arc<GPUContext>,
    solver_registry: SolverRegistry,
    pipeline: SimulationPipeline,

    parameter_buffer: wgpu::Buffer,
    velocity_temp_buffers: [wgpu::Buffer; 2],

    physical_constants: PhysicalConstants,
    current_solver: usize,
}

impl GPUSimulationEngine {
    fn switch_solver(&mut self, _solver_id: usize) {
        todo!();
    }
}

impl super::SimulationBackend for GPUSimulationEngine {
    fn new(data: &SimulationData, gpu_context: Arc<GPUContext>) -> anyhow::Result<Self> where Self: Sized {
        let mut solver_registry = SolverRegistry::new(gpu_context.clone());
        solver_registry.register(
            SolverDescriptor {
                id: 1,
                name: "Conjugate Gradient",
                create: operator::pressure_solution::construct_cg_solver_kernel,
            }
        );

        let parameter_buffer = gpu_context.device.create_buffer(
            &wgpu::BufferDescriptor{
                label: Some("Parameter buffer"),
                size: std::mem::size_of::<Parameter>() as u64,
                usage: BufferUsages::COPY_DST | BufferUsages::UNIFORM,
                mapped_at_creation: false,
            }
        );

        let velocity_temp_buffers = [
            gpu_context.device.create_buffer(
            &wgpu::BufferDescriptor{
                label: Some("Temp velx buffer"),
                size: data.velocity[0].size(),
                usage: BufferUsages::COPY_DST | BufferUsages::STORAGE,
                mapped_at_creation: false,
            }),
            gpu_context.device.create_buffer(
            &wgpu::BufferDescriptor{
                label: Some("Temp vely buffer"),
                size: data.velocity[1].size(),
                usage: BufferUsages::COPY_DST | BufferUsages::STORAGE,
                mapped_at_creation: false,
            }),
        ];
        
        let mut pipeline = SimulationPipeline::new();

        pipeline.add_step(operator::advection::construct_velocity_avection_kernel(&gpu_context.device, data, &velocity_temp_buffers, &parameter_buffer)?)
            //.add_step(operator::boundary_condition::construct_boundary_condition_kernel(&gpu_context.device, data, &parameter_buffer)?)
            //.add_step(operator::apply_internal_condition())
            .add_step(operator::divergence_computation::construct_divergence_calculator_kernel(&gpu_context.device, data, &parameter_buffer)?)
            .add_step(operator::pressure_solution::construct_cg_solver_kernel(&gpu_context.device, data, &parameter_buffer)?)
            .add_step(operator::velocity_projection::construct_velocity_projection_kernel(&gpu_context.device, data, &parameter_buffer)?)
            .add_step(operator::velocity_projection::construct_velocity_abs_val_kernel(&gpu_context.device, data, &parameter_buffer)?)
            //.add_step(operator::boundary_condition::construct_boundary_condition_kernel(&gpu_context.device, data, &parameter_buffer)?)
            .add_step(operator::divergence_computation::construct_divergence_calculator_kernel(&gpu_context.device, data, &parameter_buffer)?);
            //.add_step(operator::apply_internal_condition())
            //.add_step(operator::compute_divergence())
            //.add_step(operator::divergence_check());

        Ok(Self {
            gpu_context,
            pipeline,
            solver_registry,
            physical_constants: Default::default(),
            current_solver: 1,
            parameter_buffer,
            velocity_temp_buffers
        })
    }

    fn step(&mut self, settings: &super::data::SimulationSettings, mut frame: Frame) {
        if settings.solver_id != self.current_solver {
            self.switch_solver(settings.solver_id);
        }

        match self.pipeline.modify_solver_settings(&settings.solver_settings) {
            Err(msg) => tracing::error!("{}", msg),
            Ok(()) => {},
        };
        
        let dt = 0.01f32;

        let dim_x = frame.data.grid.get_dimensions()[0];
        let dim_y = frame.data.grid.get_dimensions()[1];
        let cell_num = dim_x * dim_y;
        
        let parameter = Parameter {
                grid_size: frame.data.grid.get_dimensions().map(|d| d as u32),
                dx: frame.data.grid.get_delta_x(),
                density: self.physical_constants.density,
                dt,
                padding: [0; 2]
            };

        self.gpu_context.queue.write_buffer(
            &self.parameter_buffer,
            0,
            bytemuck::bytes_of(&parameter)
        );

        frame.command_encoder.copy_buffer_to_buffer(
            &frame.data.velocity[0], 0, 
            &self.velocity_temp_buffers[0], 0, 
            frame.data.velocity[0].size()
        );
        frame.command_encoder.copy_buffer_to_buffer(
            &frame.data.velocity[1], 0, 
            &self.velocity_temp_buffers[1], 0, 
            frame.data.velocity[1].size()
        );

        self.gpu_context.queue.write_buffer(
            &frame.data.is_solid,
            0,
            bytemuck::cast_slice(&is_solid_vec(dim_x, dim_y))
        );

        match self.pipeline.run(&mut frame) {
            Err(msg) => tracing::error!("Could not execute pipeline: {}", msg),
            Ok(_time) => {},// println!("{}", format!("Pipeline succeeded after {} micros", time).green()),
        };
        return;
    }

    fn get_solver_names(&self) -> HashMap<usize, &'static str> {
        self.solver_registry.get_name_map()
    }

    fn get_device(&self) -> super::BackendDevice {
        super::BackendDevice::GPU
    }
}

fn is_solid_vec(dim_x: usize, dim_y: usize) -> Vec<u32> {
    let cell_num = dim_x * dim_y;
    let mut result = vec![0 as u32; cell_num];
    for x in 0..dim_x {
        result[x] = 1;
    }
    for x in 0..dim_x {
        result[(dim_y - 1) * dim_x + x] = 1;
    }
    result
}