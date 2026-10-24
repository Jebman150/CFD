use std::{collections::HashMap, sync::Arc};

use anyhow::Error;
use wgpu::Device;

use crate::{app::gpu_context::GPUContext, simulation::{data::SimulationData, gpu_backend::pipeline::operator::SimulationOperator}};

#[derive(Debug)]
pub struct SolverDescriptor {
    pub id: usize,
    pub name: &'static str,

    pub create: fn(&Arc<Device>, &SimulationData, &wgpu::Buffer) -> anyhow::Result<Box<dyn SimulationOperator>>,
}

pub struct SolverRegistry {
    gpu_context: Arc<GPUContext>,
    solvers: HashMap<usize, SolverDescriptor>,
}

impl SolverRegistry {
    pub fn new(gpu_context: Arc<GPUContext>) -> Self {
        Self {
            gpu_context,
            solvers: HashMap::new(),
        }
    }

    pub fn register(&mut self, solver_desc: SolverDescriptor) {
        tracing::info!("Registered solver: {:?}", solver_desc);
        self.solvers.insert(solver_desc.id, solver_desc);
    }

    pub fn create_solver(&mut self, id: usize, data: &SimulationData, parameter: &wgpu::Buffer) -> anyhow::Result<Box<dyn SimulationOperator>> {
        match self.solvers.get(&id) {
            Some(desc) => Ok((desc.create)(&self.gpu_context.device, data, parameter)?),
            None => Err(Error::msg(format!("Could not find solver with id {}", id))),
        }
    }

    pub fn get_solver_name(&self, id: usize) -> Result<&'static str, String> {
        match self.solvers.get(&id) {
            Some(solver) => Ok(solver.name),
            None => Err(format!("Could not find solver with id {}", id)),
        }
    }

    pub fn get_name_map(&self) -> HashMap<usize, &'static str> {
        let mut result = HashMap::new();
        for (id, desc) in &self.solvers {
            result.insert(*id, desc.name);
        }
        return result;
    }
}