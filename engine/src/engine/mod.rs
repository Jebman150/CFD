use std::sync::Arc;

use anyhow::Error;
use tracing::instrument;
use wgpu::{CommandEncoder, CommandEncoderDescriptor};

use crate::{GPUContext, SimulationState, engine::operations::KernelOperation, gpu::kernel::ComputeKernel};

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
        tracing::debug!("Creating new simulation engine");
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
        tracing::debug!("Validating engine...");
        for k in &self.operations {
            k.validate(state)?;
        }
        tracing::debug!("Validation successfull!");
        self.is_validated = true;
        Ok(())
    }

    pub fn build_pipeline(&mut self, state: &SimulationState) -> anyhow::Result<()> {
        if !self.is_validated {
            self.validate(state)?;
        }

        for operation in &self.operations {
            self.pipeline.push(
                operation.create_kernel(&self.gpu_context, state)?
            );
        }

        Ok(())
    }

    pub fn execute_kernel(&self, state: &SimulationState, kernel_id: KernelID) -> anyhow::Result<()> {
        tracing::debug!("Executing kernel {:?}", kernel_id);
        let Some(kernel) = self.pipeline.get(kernel_id.0) else {
            return Err(Error::msg(format!("Could not find kernel with id {:?}", kernel_id)));
        };

        let mut encoder = self.gpu_context.device.create_command_encoder(&CommandEncoderDescriptor{
            label: Some(&format!("Single kernel ({:?}) execution encoder", kernel_id)),
        });

        kernel.update_resources(state, &mut encoder);
        kernel.encode(&mut encoder)?;

        self.gpu_context.queue.submit([encoder.finish()]);
        tracing::debug!("Done");
        Ok(())
    }

    pub fn encode(&self, state: &SimulationState, encoder: &mut CommandEncoder) -> anyhow::Result<()> {
        for k in &self.pipeline {
            k.update_resources(state, encoder);

            k.encode(encoder)?;
        }
        Ok(())
    }
}