use std::sync::Arc;

use engine::{GPUContext, ParameterLayout, SIUnitDimensions, ScatteredVectorAdvectionOperator, VectorFieldStorage};
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
fn test_engine_scattered_vector_advection(
    #[from(crate::gpu_context)] gpu_context: Arc<GPUContext>,
    #[case] grid_size: [usize; 2],
    #[case] seed: u64
) -> anyhow::Result<()> {
    common::with_test_tracing(&format!("{}{:?}", function_name!(), grid_size), || {
        use crate::common::assertions;

        let [nx, ny] = grid_size;
        tracing::info!("Testing scattered vector advection kernel for case [{}, {}], seed {}", nx, ny, seed);

        let mut engine = common::setup_default_engine(&gpu_context)?;

        let (mut state, _pressure, vel, grid_dim_x, grid_dim_y, dx, dy) =
            common::setup_minimal_grid(&gpu_context, grid_size)?;


        let delta_t = state.parameter_mut().register_f32(0.01, SIUnitDimensions::TIME);

        let (u, v) = random_init::generate_random_mac_velocity(grid_size, seed);

        match state.get_vector_field(vel)? {
            VectorFieldStorage::Scattered(fields) => {
                fields[0].write_buffer(&gpu_context.queue, &u);
                fields[1].write_buffer(&gpu_context.queue, &v);
            }
            _ => anyhow::bail!("Expected scattered velocity field"),
        }

        let dx_value = state.parameter_ref().get_param_f32(dx)?;
        let dy_value = state.parameter_ref().get_param_f32(dy)?;
        let dt_value = state.parameter_ref().get_param_f32(delta_t)?;

        let expected = cpu_validator::vector_advection_cpu(
            &u, &v,
            &u, &v,
            grid_size,
            dx_value, dy_value, dt_value
        );

        let adv_kernel = engine.add_operation(
            ScatteredVectorAdvectionOperator {
                vector_field: vel,
                advected_field: vel,
                parameter_layout: ParameterLayout::new(
                    vec![dx, dy, delta_t],
                    vec![grid_dim_x, grid_dim_y],
                ),
            },
        );

        engine.validate(&state)?;
        engine.build_pipeline(&state)?;
        engine.execute_kernel(&state, adv_kernel)?;

        let actual = 
        match state.get_vector_field(vel)? {
            VectorFieldStorage::Scattered(fields) => (
                fields[0].read_buffer(&gpu_context.device, &gpu_context.queue).block_on()?,
                fields[1].read_buffer(&gpu_context.device, &gpu_context.queue).block_on()?
            ),
            _ => anyhow::bail!("Expected scattered velocity field"),
        };

        assertions::assert_approximately_equal(
            &format!("vec_adv_x_{:?}", grid_size),
            &actual.0,
            &expected.0,
            1e-3,
            1e-4,
        );

        assertions::assert_approximately_equal(
            &format!("vec_adv_y_{:?}", grid_size),
            &actual.1,
            &expected.1,
            1e-3,
            1e-4,
        );

        Ok(())
    })
}