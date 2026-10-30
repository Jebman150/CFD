use std::sync::Arc;

use anyhow::Error;
use tracing::instrument;
use wgpu::{CommandBuffer, CommandEncoderDescriptor};

use crate::{GPUContext, SimulationState, engine::operations::KernelOperation, gpu::kernel::{self, ComputeKernel}};

pub mod operations;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KernelID(pub usize);

pub struct SimulationEngine {
    gpu_context: Arc<GPUContext>,
    operations: Vec<Box<dyn KernelOperation>>,
    is_validated: bool,
    pipeline: Vec<Box<dyn ComputeKernel>>,
}

impl SimulationEngine {
    #[instrument]
    pub fn new(gpu_context: &Arc<GPUContext>) -> Self {
        tracing::info!("Creating new simulation engine");
        Self {
            gpu_context: gpu_context.clone(),
            operations: Vec::new(),
            is_validated: false,
            pipeline: Vec::new(),
        }
    }

    pub fn add_operation<K>(&mut self, op: K) -> KernelID 
    where 
        K: KernelOperation + 'static
    {
        self.operations.push(Box::new(op));
        KernelID(self.operations.len() - 1)
    }

    pub fn validate(&mut self, state: &SimulationState) -> anyhow::Result<()> {
        tracing::info!("Validating engine...");
        for k in &self.operations {
            k.validate(state)?;
        }
        tracing::info!("Validation successfull!");
        self.is_validated = true;
        Ok(())
    }

    pub fn build_pipeline(&mut self, state: &SimulationState) -> anyhow::Result<()> {
        if !self.is_validated {
            self.validate(state)?;
        }

        for operation in &self.operations {
            self.pipeline.push(
                operation.create_kernel(&self.gpu_context.device, state)?
            );
        }

        Ok(())
    }

    pub fn execute_kernel(&self, state: &SimulationState, kernel_id: KernelID) -> anyhow::Result<()> {
        tracing::info!("Executing kernel {:?}", kernel_id);
        let Some(kernel) = self.pipeline.get(kernel_id.0) else {
            return Err(Error::msg(format!("Could not find kernel with id {:?}", kernel_id)));
        };

        let Some(parameter_layout) = self.operations.get(kernel_id.0).map(|op| op.get_param_layout()) else {
            return Err(Error::msg(format!("Could not find parameter layout for kernel with id {:?}", kernel_id)));
        };

        let params = state.parameter_ref();

        let param_u32_vec = parameter_layout.get_u32_params().iter()
            .map(|id| params.get_param_u32(id.clone()).unwrap())
            .collect::<Vec<u32>>();

        let param_f32_vec = parameter_layout.get_f32_params().iter()
            .map(|id| params.get_param_f32(id.clone()).unwrap())
            .collect::<Vec<f32>>();

        let buffers = kernel.get_helper_buffer();
        let u32_buffer = &buffers[0];
        let f32_buffer = &buffers[1];

        self.gpu_context.queue.write_buffer(
            u32_buffer,
            0,
            bytemuck::cast_slice(&param_u32_vec)
        );

        self.gpu_context.queue.write_buffer(
            f32_buffer,
            0,
            bytemuck::cast_slice(&param_f32_vec)
        );

        // update parameters

        let mut encoder = self.gpu_context.device.create_command_encoder(&CommandEncoderDescriptor{
            label: Some(&format!("Single kernel ({:?}) execution encoder", kernel_id)),
        });
        kernel.encode(&mut encoder)?;

        self.gpu_context.queue.submit([encoder.finish()]);
        tracing::info!("Done");
        Ok(())
    }
}