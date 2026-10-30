use std::fs::OpenOptions;
use std::sync::Arc;

use engine::GPUContext;
use engine::ParamID;
use engine::SIUnitDimensions;
use engine::ScalarFieldID;
use engine::SimulationEngine;
use engine::SimulationState;
use engine::VectorFieldID;
use pollster::FutureExt;
use tracing_subscriber::Layer;
use tracing_subscriber::layer::SubscriberExt;

pub mod random_init;
pub mod assertions;

pub fn setup_default_engine(gpu_context: &Arc<GPUContext>) -> anyhow::Result<SimulationEngine> {
    let engine = SimulationEngine::new(gpu_context);

    return Ok(engine);
}

pub fn setup_empty_grid(gpu_context: &Arc<GPUContext>) -> anyhow::Result<SimulationState> {
    SimulationState::new(gpu_context)
}

pub fn setup_minimal_grid(
    gpu_context: &Arc<GPUContext>,
    size: [usize; 2]
) -> anyhow::Result<(
        SimulationState,
        ScalarFieldID,
        VectorFieldID,
        ParamID,
        ParamID,
        ParamID,
        ParamID
    )> {
    let mut state = setup_empty_grid(gpu_context)?;
    let pressure_field = state.add_scalar_field(SIUnitDimensions::PRESSURE, size);
    let velocity_field = state.add_scattered_vector_field(SIUnitDimensions::VELOCITY, size);

    let params = state.parameter_mut();
    let grid_size_x = params.register_u32(size[0] as u32, SIUnitDimensions::NO_UNIT);
    let grid_size_y = params.register_u32(size[1] as u32, SIUnitDimensions::NO_UNIT);

    let grid_dx = params.register_f32(1.0 / size[0] as f32, SIUnitDimensions::LENGTH);
    let grid_dy = params.register_f32(1.0 / size[1] as f32, SIUnitDimensions::LENGTH);

    Ok((state, pressure_field, velocity_field, grid_size_x, grid_size_y, grid_dx, grid_dy))
}

pub fn setup_gpu_context() -> anyhow::Result<GPUContext> {
    GPUContext::new().block_on()
}

pub fn with_test_tracing<T>(
    log_name: &str,
    test: impl FnOnce() -> T,
) -> T {
    std::fs::create_dir_all("tests/log").unwrap();

    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(format!("tests/log/{log_name}.log"))
        .unwrap();

    let subscriber = tracing_subscriber::registry()
        .with(
            tracing_subscriber::fmt::layer()
                .with_ansi(false)
                .with_writer(file),
        )
        .with(
            tracing_subscriber::fmt::layer()
                .with_filter(tracing_subscriber::filter::LevelFilter::INFO),
        );

    tracing::subscriber::with_default(subscriber, test)
}
