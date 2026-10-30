mod common;
mod cpu_validator;
mod kernels;
use std::sync::Arc;

use engine::GPUContext;
use function_name::named;
use rstest::*;

#[fixture]
fn gpu_context() -> Arc<GPUContext> {
    Arc::new(common::setup_gpu_context().unwrap())
}

#[named]
#[rstest]
fn test_engine_state_setup() -> anyhow::Result<()> {
    common::with_test_tracing(function_name!(), || {
        let gpu_context = Arc::new(common::setup_gpu_context()?);
        let _engine = common::setup_default_engine(&gpu_context)?;

        let test_grid_size = [10, 10];
        let (_state, _pressure, _vel, _grid_x, _grid_y, _dx, _dy) = common::setup_minimal_grid(&gpu_context, test_grid_size)?;

        Ok(())
    })
}