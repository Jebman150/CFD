use std::{collections::HashMap, sync::Arc};

use crate::{app::gpu_context::GPUContext, simulation::{BackendDevice, data::{Frame, ScalarField, SimulationData}}};

use super::data::{
    PhysicalConstants,
    SimulationSettings
};

use data::CPUSimulationData;

use pipeline::{
    SimulationPipeline,
    operator::{
        self,
        solver_registry::{
            SolverRegistry,
            SolverDescriptor
        }
    }
};
use wgpu::Queue;

mod data;
mod pipeline;

pub struct CPUSimulationEngine {
    queue: Arc<Queue>,
    data: CPUSimulationData,
    solver_registry: SolverRegistry,
    pipeline: SimulationPipeline,

    physical_constants: PhysicalConstants,
    current_solver: usize,
}

impl CPUSimulationEngine {
    fn switch_solver(&mut self, solver_id: usize) {
        match self.solver_registry.create_solver(solver_id) {
            Err(msg) => tracing::error!("{}", msg),
            Ok(new_solver) => {
                match self.pipeline.set_solver(new_solver) {
                    Err(msg) => tracing::error!("{}", msg),
                    Ok(()) => {},
                };
                self.current_solver = solver_id;
                tracing::info!("Changed solver to {}", self.solver_registry.get_solver_name(solver_id).unwrap());
            }
        }
    }

    fn prepare_data_for_rendering(
        &mut self,
        state: &mut super::data::SimulationData,
        scalar_field: ScalarField
    ) {
        if let Some(dest) = state.scalar_fields.get(&scalar_field) {
            if let Some(source) = match scalar_field {
                ScalarField::Pressure => Some(&self.data.pressure.data),
                ScalarField::Divergence => Some(&self.data.divergence.data),
                ScalarField::Velocity => Some(&self.data.abs_velocity.data),
                /*
                _ => {
                    tracing::warn!("{} not implemented for rendering. Choose another.", <&str>::from(scalar_field));
                    None
                }*/
            } {
                self.queue.write_buffer(
                    dest,
                    0,
                    bytemuck::cast_slice(source)
                );
            }
        }
        
        self.queue.write_buffer(
            &state.is_solid,
            0,
            bytemuck::cast_slice(&self.data.is_solid.data)
        );

        self.queue.write_buffer(
            &state.velocity[0],
            0,
            bytemuck::cast_slice(&self.data.velocity[0].data)
        );
        self.queue.write_buffer(
            &state.velocity[1],
            0,
            bytemuck::cast_slice(&self.data.velocity[1].data)
        );
    }
}

impl super::SimulationBackend for CPUSimulationEngine {
    fn new(data: &SimulationData, gpu_context: Arc<GPUContext>) -> anyhow::Result<Self> {
        let mut solver_registry = SolverRegistry::new();
        solver_registry.register(
            SolverDescriptor {
                id: 1,
                name: "Conjugate Gradient",
                create: operator::pressure_solution::cg_solver
            }
        );
        solver_registry.register(
            SolverDescriptor {
                id: 2,
                name: "Gauss Seidel",
                create: operator::pressure_solution::gs_solver
            }
        );

        let mut data = CPUSimulationData::new(data.grid.get_dimensions());
        operator::set_solid_cells_startup(&mut data);
        
        let mut pipeline = SimulationPipeline::new(
            operator::pressure_solution::cg_solver()
        );
        let solver_ref = pipeline.solver_ref();

        pipeline.add_step(operator::advect_velocities())
            .add_step(operator::apply_boundary_condition())
            .add_step(operator::apply_internal_condition())
            .add_step(operator::compute_divergence())
            .add_step(operator::pressure_solver(solver_ref))
            .add_step(operator::project_velocities())
            .add_step(operator::velocity_abs_val_calculation())
            .add_step(operator::apply_internal_condition())
            .add_step(operator::compute_divergence())
            .add_step(operator::divergence_check());


        Ok(Self {
            queue: gpu_context.queue.clone(),
            data: data,
            pipeline,
            solver_registry,
            physical_constants: Default::default(),
            current_solver: 1,
        })
    }

    fn step(&mut self, settings: &SimulationSettings, frame: Frame) {
        if settings.solver_id != self.current_solver {
            self.switch_solver(settings.solver_id);
        }

        match self.pipeline.modify_solver_settings(&settings.solver_settings) {
            Err(msg) => tracing::error!("{}", msg),
            Ok(()) => {},
        };
        
        let dt = 0.01f32;

        match self.pipeline.run(&mut self.data, &self.physical_constants, dt) {
            Err(msg) => tracing::error!("Could not execute pipeline: {}", msg),
            Ok(_time) => {},// println!("{}", format!("Pipeline succeeded after {} micros", time).green()),
        };

        self.prepare_data_for_rendering(frame.data, frame.rendered_field);
    }

    fn get_solver_names(&self) -> HashMap<usize, &'static str> {
        self.solver_registry.get_name_map()
    }

    fn get_device(&self) -> BackendDevice {
        BackendDevice::CPU
    }

    
}