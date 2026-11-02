use std::{collections::HashMap, sync::Arc};

use wgpu::CommandEncoder;

use engine::{
    GPUContext, ParamID, ScalarFieldID, SimulationEngine, SimulationState, VectorFieldID
};

mod init;
mod settings;

pub use settings::SimulationSettings;

#[derive(Debug, Clone, strum_macros::IntoStaticStr, strum_macros::EnumIter, PartialEq, Eq)]
pub enum BackendDevice {
    None,
    CPU,
    GPU,
}

pub struct SimulationHandles {
    pressure: ScalarFieldID,
    velocity: VectorFieldID,
    divergence: ScalarFieldID,

    abs_velocity: ScalarFieldID,

    grid_size_x: ParamID,
    grid_size_y: ParamID,
    dx: ParamID,
    dy: ParamID,
    dt: ParamID,
}

pub struct Simulation {
    gpu_context: Arc<GPUContext>,
    settings: SimulationSettings,
    state: SimulationState,
    handles: SimulationHandles,
    engine: SimulationEngine,
}

impl Simulation {
    pub fn new(gpu_context: Arc<GPUContext>) -> anyhow::Result<Self> {
        let default_settings = SimulationSettings::default();
        let (state, handles) = init::initialize_state(&gpu_context, default_settings.grid_size);
        let engine = init::initialize_engine(&gpu_context, &state, &handles).unwrap();
        Ok(Self {
            state,
            handles,
            engine,
            settings: default_settings,
            gpu_context,
        })
    }

    pub fn update(&mut self, encoder: &mut CommandEncoder) {
        if !self.settings.is_running && !self.settings.step {
            return;
        }

        self.engine.encode(&self.state, encoder).unwrap();
    }

    pub fn settings_mut(&mut self) -> &mut SimulationSettings {
        &mut self.settings
    }

    pub fn settings(&self) -> &SimulationSettings {
        &self.settings
    }

    pub fn state_ref(&self) -> &SimulationState {
        &self.state
    }

    pub fn get_solver_names(&self) -> HashMap<usize, &'static str> {
        return HashMap::new();
    }
}