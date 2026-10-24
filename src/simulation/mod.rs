
use std::{collections::HashMap, sync::Arc};

use data::{
    SimulationData,
    SimulationSettings,
    grid::Grid2D,
    ScalarField,
    Frame
};
use wgpu::CommandEncoder;

use crate::app::gpu_context::GPUContext;
use gpu_backend::GPUSimulationEngine;
use cpu_backend::CPUSimulationEngine;

pub mod cpu_backend;
pub mod gpu_backend;

pub mod data;

trait SimulationBackend
{
    fn new(data: &SimulationData, gpu_context: Arc<GPUContext>) -> anyhow::Result<Self> where Self: Sized;
    fn step(&mut self, settings: &SimulationSettings, frame: Frame);

    fn get_solver_names(&self) -> HashMap<usize, &'static str>;
    fn get_device(&self) -> BackendDevice;
}

#[derive(Debug, Clone, strum_macros::IntoStaticStr, strum_macros::EnumIter, PartialEq, Eq)]
pub enum BackendDevice {
    None,
    CPU,
    GPU,
}

pub struct Simulation {
    gpu_context: Arc<GPUContext>,
    settings: SimulationSettings,
    state: SimulationData,
    pub grid: Grid2D,
    backend: Option<Box<dyn SimulationBackend>>,
}

impl Simulation {
    pub fn new(gpu_context: Arc<GPUContext>) -> Self {
        let default_settings = SimulationSettings::default();

        Self {
            state: SimulationData::new(&gpu_context, default_settings.grid_size),
            grid: Grid2D::new(default_settings.grid_size, default_settings.dx),
            settings: default_settings,
            backend: None,
            gpu_context,
        }
    }

    pub fn switch_to_gpu_backend(&mut self) -> anyhow::Result<()> {
        if let Some(existing_backend) = &self.backend {
            if existing_backend.get_device() == BackendDevice::GPU {
                return Ok(())
            }
        }

        self.backend = Some(Box::new(
            GPUSimulationEngine::new(&self.state, self.gpu_context.clone())?
        ));
        tracing::info!("Switched to GPU");

        return Ok(());
    }

    pub fn switch_to_cpu_backend(&mut self) -> anyhow::Result<()> {
        if let Some(existing_backend) = &self.backend {
            if existing_backend.get_device() == BackendDevice::CPU {
                return Ok(())
            }
        }

        self.backend = Some(Box::new(
            CPUSimulationEngine::new(&self.state, self.gpu_context.clone())?
        ));
        tracing::info!("Switched to CPU");

        return Ok(());
    }

    pub fn update(&mut self, encoder: &mut CommandEncoder, rendered_field: ScalarField) {
        if !self.settings.is_running && !self.settings.step {
            return;
        }

        let result = match self.settings.device {
            BackendDevice::CPU => self.switch_to_cpu_backend(),
            BackendDevice::GPU => self.switch_to_gpu_backend(),
            BackendDevice::None => {self.backend = None; return;},
        };
        if result.is_err() {
            tracing::error!("Could not use specified device {}: {:#?}", <&str>::from(self.settings.device.clone()), result);
            return;
        }

        let Some(backend) = self.backend.as_mut() else {
            return;
        };

        let frame = Frame {
            data: &mut self.state,
            command_encoder: encoder,
            rendered_field: rendered_field,
        };

        backend.step(&self.settings, frame);
    }

    pub fn settings_mut(&mut self) -> &mut SimulationSettings {
        &mut self.settings
    }

    pub fn settings(&self) -> &SimulationSettings {
        &self.settings
    }

    pub fn state_ref(&self) -> &SimulationData {
        &self.state
    }

    pub fn get_solver_names(&self) -> HashMap<usize, &'static str> {
        let Some(existing_backend) = &self.backend else {
            return HashMap::new();
        };
        existing_backend.get_solver_names()
    }
}