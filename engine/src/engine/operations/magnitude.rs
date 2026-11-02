use std::sync::Arc;

use anyhow::Error;

use crate::{GPUContext, ParameterLayout, SIUnitDimensions, ScalarFieldID, SimulationState, VectorFieldID, VectorFieldStorage::Scattered, engine::operations::{KernelOperation, parameter_validation::{ParameterRequirement, ParameterRequirementLayout}}, gpu::{init_helper, kernel::{ComputeKernel, LeafKernel}, kernel_resources::{BufferCopy, ParameterResource}}};



pub struct VectorMagnitudeOperator {
    pub input: VectorFieldID,
    pub output: ScalarFieldID,
    pub parameter_layout: ParameterLayout,
}

impl KernelOperation for VectorMagnitudeOperator {
    fn name(&self) -> &'static str {
        "Vector magnitude operator"
    }

    fn validate(&self, state: &SimulationState) -> anyhow::Result<()> {
        let input_meta = state.get_vector_field_metadata(self.input)?;
        let output_meta = state.get_scalar_field_metadata(self.output)?;

        if input_meta.si_unit != output_meta.si_unit {
            return Err(Error::msg(format!("Validation error: {} != {} in magnitude calculation", input_meta.si_unit, output_meta.si_unit)));
        }

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
            f32_requirements: vec![],
        };

        parameter_requirements.validate(state.parameter_ref(), &self.parameter_layout)?;

        Ok(())
    }
    
    fn create_kernel(&self, gpu_context: &Arc<GPUContext>, state: &SimulationState) -> anyhow::Result<Box<dyn ComputeKernel>> {
        let Scattered([vec_x, vec_y]) = state.get_vector_field(self.input)? else {
            return Err(Error::msg("Expected scattered vector field for magnitude calculation kernel. Got collocated vector field"));
        };

        let output = state.get_scalar_field(self.output)?;

        let mut layout_buffer_entries = Vec::new();
        init_helper::add_layout_buffer(&mut layout_buffer_entries, 0, true);
        init_helper::add_layout_buffer(&mut layout_buffer_entries, 1, true);
        init_helper::add_layout_buffer(&mut layout_buffer_entries, 2, false);
        init_helper::add_layout_buffer(&mut layout_buffer_entries, 3, true);

        let parameters = ParameterResource::new(gpu_context, self.parameter_layout.clone());

        Ok(Box::new(LeafKernel::new("Projection")
            .build_pipeline(
                &gpu_context.device,
                "shaders/projection.wgsl",
                "projection",
                &layout_buffer_entries,
                &vec![
                    (0, vec_x.get_buffer_ref()),
                    (1, vec_y.get_buffer_ref()),
                    (2, output.get_buffer_ref()),
                    (3, &parameters.get_u32_buffer_ref())
                ],
                output.get_buffer_ref().size() / std::mem::size_of::<f32>() as u64
            )?
            .add_parameter_resource(parameters)
        ))
    }
    
    fn get_param_layout(&self) -> &ParameterLayout {
        &self.parameter_layout
    }    
}