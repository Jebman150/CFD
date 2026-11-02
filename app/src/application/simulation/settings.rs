use crate::application::simulation::BackendDevice;



pub struct PhysicalConstants {
    pub density: f32,
}

impl Default for PhysicalConstants {
    fn default() -> Self {
        Self { density: 1.0 }
    }
}

#[derive(Debug)]
pub struct SolverSettings {
    pub maxit: usize,
    pub tolerance: f32,
}

impl Default for SolverSettings {
    fn default() -> Self {
        Self {
            maxit: 1000,
            tolerance: 0.005
        }
    }
}

#[derive(Debug)]
pub struct SimulationSettings {
    pub grid_size: [usize; 2],
    pub dx: [f32; 2],
    pub is_running: bool,
    pub step: bool,
    pub solver_id: usize,
    pub solver_settings: SolverSettings,
    pub device: BackendDevice,
}

impl Default for SimulationSettings {
    fn default() -> Self {
        Self {
            grid_size: [50, 50],
            dx: [1.0/50.0, 1.0/50.0],
            solver_id: 1,
            step: Default::default(),
            is_running: false,
            solver_settings: Default::default(),
            device: BackendDevice::GPU,
        }
    }
}