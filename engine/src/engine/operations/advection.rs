use std::sync::Arc;

use anyhow::Error;
use wgpu::Device;

use crate::{ParameterLayout, SIUnitDimensions, ScalarFieldID, SimulationState, VectorFieldID, VectorFieldStorage::Scattered, engine::operations::{KernelOperation, parameter_validation::{ParameterRequirement, ParameterRequirementLayout}}, gpu::{init_helper, kernel::{ComputeKernel, LeafKernel}}, state::fields::Field};



pub struct ScalarAdvectionOperator {
    pub vector_field: VectorFieldID,
    pub advected_field: ScalarFieldID,
    pub parameter_layout: ParameterLayout,
}

impl KernelOperation for ScalarAdvectionOperator {
    fn name(&self) -> &'static str {
        "Scalar advection operator"
    }

    fn validate(&self, state: &SimulationState) -> anyhow::Result<()> {
        let input_meta = state.get_vector_field_metadata(self.vector_field)?;
        let _output_meta = state.get_scalar_field_metadata(self.advected_field)?;

        if input_meta.si_unit != SIUnitDimensions::VELOCITY {
            return Err(Error::msg(format!("Validation error: expected velocity field for scalar advection, got {}", input_meta.si_unit)));
        }
        // Output unit stays the same on advection, so no unit validation is required

        let parameter_requirements = ParameterRequirementLayout {
            u32_requirements: vec![
                ParameterRequirement {
                    name: "Grid size x",
                    units: SIUnitDimensions::NO_UNIT
                },
                ParameterRequirement {
                    name: "Grid size y",
                    units: SIUnitDimensions::NO_UNIT
                },
            ],
            f32_requirements: vec![
                ParameterRequirement {
                    name: "Grid dx",
                    units: SIUnitDimensions::LENGTH
                },
                ParameterRequirement {
                    name: "Grid dy",
                    units: SIUnitDimensions::LENGTH
                },
                ParameterRequirement {
                    name: "Delta t",
                    units: SIUnitDimensions::TIME
                },
            ],
        };

        parameter_requirements.validate(state.parameter_ref(), &self.parameter_layout)?;

        Ok(())
    }
    
    fn create_kernel(&self, device: &Arc<Device>, state: &SimulationState) -> anyhow::Result<Box<dyn ComputeKernel>> {
        let u32_params = self.parameter_layout.create_u32_buffers(device);
        let f32_params = self.parameter_layout.create_f32_buffer(device);
        

        let Scattered([vel_x, vel_y]) = state.get_vector_field(self.vector_field)? else {
            return Err(Error::msg("Expected scattered velocity field for scalar advection kernel. Got collocated vector field"));
        };

        let scalar_field = state.get_scalar_field(self.advected_field)?;

        let temp_value_buffer = init_helper::create_storage_buffer_with_capacity(
            device,
            "Advection: Scalar temporary buffer",
            scalar_field.get_buffer_ref().size()
        );

        let mut layout_buffer_entries = Vec::new();
        init_helper::add_layout_buffer(&mut layout_buffer_entries, 0, true);
        init_helper::add_layout_buffer(&mut layout_buffer_entries, 1, true);
        init_helper::add_layout_buffer(&mut layout_buffer_entries, 2, true);
        init_helper::add_layout_buffer(&mut layout_buffer_entries, 3, false);
        init_helper::add_layout_buffer(&mut layout_buffer_entries, 4, true);
        init_helper::add_layout_buffer(&mut layout_buffer_entries, 5, true);

        Ok(Box::new(LeafKernel::new("Divergence computation")
            .build_pipeline(
                device,
                "shaders/scalar_advection.wgsl",
                "scalar_advection",
                &layout_buffer_entries,
                &vec![
                    (0, vel_x.get_buffer_ref()),
                    (1, vel_y.get_buffer_ref()),
                    (2, &temp_value_buffer),
                    (3, scalar_field.get_buffer_ref()),
                    (4, &u32_params),
                    (5, &f32_params)
                ],
                scalar_field.get_buffer_ref().size() / std::mem::size_of::<f32>() as u64
            )?
            .save_helper_buffer(u32_params)
            .save_helper_buffer(f32_params)
            .save_helper_buffer(temp_value_buffer)
        ))
    }
    
    fn get_param_layout(&self) -> &ParameterLayout {
        &self.parameter_layout
    }    
}