use std::sync::Arc;

use anyhow::Error;
use wgpu::Device;

use crate::{SIUnitDimensions, ScalarFieldID, SimulationState, VectorFieldID, engine::operations::parameter_validation::{ParameterRequirement, ParameterRequirementLayout}, gpu::{init_helper, kernel::{ComputeKernel, LeafKernel, ParameterLayout}}, state::fields::{Field, VectorFieldStorage::{Collocated, Scattered}}};

mod parameter_validation;
pub mod divergence;
pub mod advection;

pub trait KernelOperation {
    fn name(&self) -> &'static str;

    fn get_param_layout(&self) -> &ParameterLayout;
    fn validate(&self, state: &SimulationState) -> anyhow::Result<()>;
    fn create_kernel(&self, device: &Arc<Device>, state: &SimulationState) -> anyhow::Result<Box<dyn ComputeKernel>>;
}

