use std::sync::Arc;

use anyhow::Error;

use crate::{GPUContext, ParameterLayout, SIUnitDimensions, ScalarFieldID, SimulationState, VectorFieldID, VectorFieldStorage::Scattered, engine::operations::{KernelOperation, parameter_validation::{ParameterRequirement, ParameterRequirementLayout}}, gpu::{init_helper, kernel::{ComputeKernel, LeafKernel}, kernel_resources::{BufferCopy, ParameterResource}}};



pub struct ProjectionOperator {
    pub pressure_field: ScalarFieldID,
    pub projected_field: VectorFieldID,
    pub parameter_layout: ParameterLayout,
}

impl KernelOperation for ProjectionOperator {
    fn name(&self) -> &'static str {
        "Scalar advection operator"
    }

    fn validate(&self, state: &SimulationState) -> anyhow::Result<()> {
        let pressure_field_meta = state.get_scalar_field_metadata(self.pressure_field)?;
        let projected_field_meta = state.get_vector_field_metadata(self.projected_field)?;

        if pressure_field_meta.si_unit != SIUnitDimensions::PRESSURE {
            return Err(Error::msg(format!("Validation error: expected pressure field for projection, got {}", pressure_field_meta.si_unit)));
        }
        if projected_field_meta.si_unit != SIUnitDimensions::VELOCITY {
            return Err(Error::msg(format!("Validation error: expected velocity field for projection, got {}", projected_field_meta.si_unit)));
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
                ParameterRequirement {
                    name: "Density",
                    units: SIUnitDimensions::DENSITY
                },
            ],
        };

        parameter_requirements.validate(state.parameter_ref(), &self.parameter_layout)?;

        Ok(())
    }
    
    fn create_kernel(&self, gpu_context: &Arc<GPUContext>, state: &SimulationState) -> anyhow::Result<Box<dyn ComputeKernel>> {
        let Scattered([vec_x, vec_y]) = state.get_vector_field(self.projected_field)? else {
            return Err(Error::msg("Expected scattered vector field for scalar advection kernel. Got collocated vector field"));
        };

        let pressure_field = state.get_scalar_field(self.pressure_field)?;

        let mut layout_buffer_entries = Vec::new();
        init_helper::add_layout_buffer(&mut layout_buffer_entries, 0, true);
        init_helper::add_layout_buffer(&mut layout_buffer_entries, 1, false);
        init_helper::add_layout_buffer(&mut layout_buffer_entries, 2, false);
        init_helper::add_layout_buffer(&mut layout_buffer_entries, 3, true);
        init_helper::add_layout_buffer(&mut layout_buffer_entries, 4, true);

        let parameters = ParameterResource::new(gpu_context, self.parameter_layout.clone());

        Ok(Box::new(LeafKernel::new("Projection")
            .build_pipeline(
                &gpu_context.device,
                "shaders/projection.wgsl",
                "projection",
                &layout_buffer_entries,
                &vec![
                    (0, pressure_field.get_buffer_ref()),
                    (1, vec_x.get_buffer_ref()),
                    (2, vec_y.get_buffer_ref()),
                    (3, &parameters.get_u32_buffer_ref()),
                    (4, &parameters.get_f32_buffer_ref())
                ],
                (vec_x.get_buffer_ref().size() + vec_y.get_buffer_ref().size()) / std::mem::size_of::<f32>() as u64
            )?
            .add_parameter_resource(parameters)
        ))
    }
    
    fn get_param_layout(&self) -> &ParameterLayout {
        &self.parameter_layout
    }    
}