use std::sync::Arc;

use engine::{GPUContext, ParameterLayout, SIUnitDimensions, SimulationEngine, SimulationState};

use crate::application::simulation::SimulationHandles;



pub fn initialize_state(gpu_context: &Arc<GPUContext>, grid_size: [usize; 2]) -> (SimulationState, SimulationHandles) {
    let mut state = engine::SimulationState::new(gpu_context).unwrap();

    let pressure = state.add_scalar_field(SIUnitDimensions::PRESSURE, grid_size);
    let divergence = state.add_scalar_field(SIUnitDimensions::div(&SIUnitDimensions::VELOCITY), grid_size);
    let abs_velocity = state.add_scalar_field(SIUnitDimensions::VELOCITY, grid_size);

    let velocity = state.add_scattered_vector_field(SIUnitDimensions::VELOCITY, grid_size);

    let params = state.parameter_mut();
    let grid_size_x = params.register_u32(grid_size[0] as u32, SIUnitDimensions::NO_UNIT);
    let grid_size_y = params.register_u32(grid_size[1] as u32, SIUnitDimensions::NO_UNIT);

    let dx = params.register_f32(1.0 / grid_size[0] as f32, SIUnitDimensions::LENGTH);
    let dy = params.register_f32(1.0 / grid_size[1] as f32, SIUnitDimensions::LENGTH);

    let dt = params.register_f32(0.01, SIUnitDimensions::TIME);

    let handles = SimulationHandles {
        pressure,
        velocity, 
        divergence,
        abs_velocity,
        grid_size_x,
        grid_size_y,
        dx,
        dy,
        dt
    };
    (state, handles)
}

pub fn initialize_engine(gpu_context: &Arc<GPUContext>, state: &SimulationState, handles: &SimulationHandles) -> anyhow::Result<SimulationEngine> {
    let mut engine = SimulationEngine::new(gpu_context);

    let _adv_kernel = engine.add_operation(
        engine::ScatteredVectorAdvectionOperator {
            vector_field: handles.velocity,
            advected_field: handles.velocity,
            parameter_layout: ParameterLayout::new(
                vec![handles.dx, handles.dy, handles.dt],
                vec![handles.grid_size_x, handles.grid_size_y],
            ),
        },
    );

    let _div_kernel = engine.add_operation(
        engine::DivergenceOperator {
            input: handles.velocity,
            output: handles.divergence,
            parameter_layout: ParameterLayout::new(
                vec![handles.dx, handles.dy],
                vec![handles.grid_size_x, handles.grid_size_y],
            ),
        },
    );

    engine.validate(&state)?;
    engine.build_pipeline(&state)?;

    Ok(engine)
}