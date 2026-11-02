use std::sync::Arc;

use anyhow::Error;

use crate::{GPUContext, ParameterLayout, SIUnitDimensions, ScalarFieldID, SimulationState, VectorFieldID, VectorFieldStorage::Scattered, engine::operations::{KernelOperation, parameter_validation::{ParameterRequirement, ParameterRequirementLayout}}, gpu::{init_helper, kernel::{ComputeKernel, LeafKernel}, kernel_resources::{BufferCopy, ParameterResource}}};



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
    
    fn create_kernel(&self, gpu_context: &Arc<GPUContext>, state: &SimulationState) -> anyhow::Result<Box<dyn ComputeKernel>> {
        let Scattered([vel_x, vel_y]) = state.get_vector_field(self.vector_field)? else {
            return Err(Error::msg("Expected scattered velocity field for scalar advection kernel. Got collocated vector field"));
        };

        let scalar_field = state.get_scalar_field(self.advected_field)?;

        let mut layout_buffer_entries = Vec::new();
        init_helper::add_layout_buffer(&mut layout_buffer_entries, 0, true);
        init_helper::add_layout_buffer(&mut layout_buffer_entries, 1, true);
        init_helper::add_layout_buffer(&mut layout_buffer_entries, 2, true);
        init_helper::add_layout_buffer(&mut layout_buffer_entries, 3, false);
        init_helper::add_layout_buffer(&mut layout_buffer_entries, 4, true);
        init_helper::add_layout_buffer(&mut layout_buffer_entries, 5, true);

        let parameters = ParameterResource::new(gpu_context, self.parameter_layout.clone());
        let scalar_field_copy = BufferCopy::new(gpu_context, scalar_field.get_buffer_ref());

        Ok(Box::new(LeafKernel::new("Divergence computation")
            .build_pipeline(
                &gpu_context.device,
                "shaders/scalar_advection.wgsl",
                "scalar_advection",
                &layout_buffer_entries,
                &vec![
                    (0, vel_x.get_buffer_ref()),
                    (1, vel_y.get_buffer_ref()),
                    (2, scalar_field_copy.get_buffer_ref()),
                    (3, scalar_field.get_buffer_ref()),
                    (4, &parameters.get_u32_buffer_ref()),
                    (5, &parameters.get_f32_buffer_ref())
                ],
                scalar_field.get_buffer_ref().size() / std::mem::size_of::<f32>() as u64
            )?
            .add_parameter_resource(parameters)
            .add_buffer_copy(scalar_field_copy)
        ))
    }
    
    fn get_param_layout(&self) -> &ParameterLayout {
        &self.parameter_layout
    }    
}



pub struct ScatteredVectorAdvectionOperator {
    pub vector_field: VectorFieldID,
    pub advected_field: VectorFieldID,
    pub parameter_layout: ParameterLayout,
}

impl KernelOperation for ScatteredVectorAdvectionOperator {
    fn name(&self) -> &'static str {
        "Scalar advection operator"
    }

    fn validate(&self, state: &SimulationState) -> anyhow::Result<()> {
        let input_meta = state.get_vector_field_metadata(self.vector_field)?;
        let _output_meta = state.get_vector_field_metadata(self.advected_field)?;

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
    
    fn create_kernel(&self, gpu_context: &Arc<GPUContext>, state: &SimulationState) -> anyhow::Result<Box<dyn ComputeKernel>> {
        let Scattered([vel_x, vel_y]) = state.get_vector_field(self.vector_field)? else {
            return Err(Error::msg("Expected scattered velocity field for scalar advection kernel. Got collocated vector field"));
        };

        let Scattered([vector_field_x, vector_field_y]) = state.get_vector_field(self.advected_field)? else {
            return Err(Error::msg("Expected scattered vector field for vector advection kernel. Got collocated vector field"));
        };

        let mut layout_buffer_entries = Vec::new();
        init_helper::add_layout_buffer(&mut layout_buffer_entries, 0, true);
        init_helper::add_layout_buffer(&mut layout_buffer_entries, 1, true);
        init_helper::add_layout_buffer(&mut layout_buffer_entries, 2, true);
        init_helper::add_layout_buffer(&mut layout_buffer_entries, 3, true);
        init_helper::add_layout_buffer(&mut layout_buffer_entries, 4, false);
        init_helper::add_layout_buffer(&mut layout_buffer_entries, 5, false);
        init_helper::add_layout_buffer(&mut layout_buffer_entries, 6, true);
        init_helper::add_layout_buffer(&mut layout_buffer_entries, 7, true);

        let parameters = ParameterResource::new(gpu_context, self.parameter_layout.clone());
        let vector_field_x_copy = BufferCopy::new(gpu_context, vector_field_x.get_buffer_ref());
        let vector_field_y_copy = BufferCopy::new(gpu_context, vector_field_y.get_buffer_ref());
        let velocity_x_copy = BufferCopy::new(gpu_context, vel_x.get_buffer_ref());
        let velocity_y_copy = BufferCopy::new(gpu_context, vel_y.get_buffer_ref());

        Ok(Box::new(LeafKernel::new("Divergence computation")
            .build_pipeline(
                &gpu_context.device,
                "shaders/vector_advection.wgsl",
                "vector_advection",
                &layout_buffer_entries,
                &vec![
                    (0, velocity_x_copy.get_buffer_ref()),
                    (1, velocity_y_copy.get_buffer_ref()),
                    (2, vector_field_x_copy.get_buffer_ref()),
                    (3, vector_field_y_copy.get_buffer_ref()),
                    (4, vector_field_x.get_buffer_ref()),
                    (5, vector_field_y.get_buffer_ref()),
                    (6, &parameters.get_u32_buffer_ref()),
                    (7, &parameters.get_f32_buffer_ref())
                ],
                (vector_field_x.get_buffer_ref().size() + vector_field_y.get_buffer_ref().size()) / std::mem::size_of::<f32>() as u64
            )?
            .add_parameter_resource(parameters)
            .add_buffer_copy(vector_field_x_copy)
            .add_buffer_copy(vector_field_y_copy)
            .add_buffer_copy(velocity_x_copy)
            .add_buffer_copy(velocity_y_copy)
        ))
    }
    
    fn get_param_layout(&self) -> &ParameterLayout {
        &self.parameter_layout
    }    
}