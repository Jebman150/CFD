use std::sync::Arc;

use crate::{GPUContext, SimulationState, gpu::kernel::{ComputeKernel, ParameterLayout}};

mod parameter_validation;
pub mod divergence;
pub mod advection;
pub mod projection;
pub mod magnitude;

pub trait KernelOperation {
    fn name(&self) -> &'static str;

    fn get_param_layout(&self) -> &ParameterLayout;
    fn validate(&self, state: &SimulationState) -> anyhow::Result<()>;
    fn create_kernel(&self, gpu_context: &Arc<GPUContext>, state: &SimulationState) -> anyhow::Result<Box<dyn ComputeKernel>>;
}

