use std::sync::Arc;

use engine::{DivergenceOperator, GPUContext, ParameterLayout, SIUnitDimensions, VectorFieldStorage};
use function_name::named;
use pollster::FutureExt;
use rstest::*;

use crate::{common::{self, random_init}, cpu_validator};


#[named]
#[rstest]
#[case([1, 1], 42)]
#[case([2, 2], 123)]
#[case([10, 10], 456)]
#[case([31, 17], 789)]
#[case([64, 64], 1337)]
#[case([128, 256], 9999)]
#[case([512, 512], 54321)]
fn test_engine_compute_divergence(
    #[from(crate::gpu_context)] gpu_context: Arc<GPUContext>,
    #[case] grid_size: [usize; 2],
    #[case] seed: u64
) -> anyhow::Result<()> {
    common::with_test_tracing(function_name!(), || {
        use crate::common::assertions;

        let [nx, ny] = grid_size;
        tracing::info!("Testing divergence kernel for case [{}, {}], seed {}", nx, ny, seed);

        let mut engine = common::setup_default_engine(&gpu_context)?;

        let (mut state, _pressure, vel, grid_dim_x, grid_dim_y, dx, dy) =
            common::setup_minimal_grid(&gpu_context, grid_size)?;

        let (u, v) = random_init::generate_random_mac_velocity(grid_size, seed);

        // Initialize the two MAC velocity components.
        match state.get_vector_field(vel)? {
            VectorFieldStorage::Scattered(fields) => {
                fields[0].write_buffer(&gpu_context.queue, &u);
                fields[1].write_buffer(&gpu_context.queue, &v);
            }
            _ => anyhow::bail!("Expected scattered velocity field"),
        }

        // Expected CPU result.
        let dx_value = state.parameter_ref().get_param_f32(dx)?;
        let dy_value = state.parameter_ref().get_param_f32(dy)?;

        let expected = cpu_validator::divergence_cpu(
            &u,
            &v,
            grid_size,
            dx_value,
            dy_value,
        );

        let divergence = state.add_scalar_field(
            SIUnitDimensions::div(&SIUnitDimensions::VELOCITY),
            grid_size,
        );

        let div_kernel = engine.add_operation(
            DivergenceOperator {
                input: vel,
                output: divergence,
                parameter_layout: ParameterLayout::new(
                    vec![dx, dy],
                    vec![grid_dim_x, grid_dim_y],
                ),
            },
        );

        engine.validate(&state)?;
        engine.build_pipeline(&state)?;
        engine.execute_kernel(&state, div_kernel)?;

        // GPU readback.
        let actual = state
            .get_scalar_field(divergence)?
            .read_buffer(&gpu_context.device, &gpu_context.queue)
            .block_on()?;

        assertions::assert_approximately_equal(
            &actual,
            &expected,
            1e-5,
            1e-4,
        );

        Ok(())
    })
}