use std::sync::Arc;

use anyhow::Error;
use wgpu::Device;

use crate::{ParameterLayout, SIUnitDimensions, ScalarFieldID, SimulationState, VectorFieldID, VectorFieldStorage::{Collocated, Scattered}, engine::operations::{KernelOperation, parameter_validation::{ParameterRequirement, ParameterRequirementLayout}}, gpu::{init_helper, kernel::{ComputeKernel, LeafKernel}}, state::fields::Field};



pub struct DivergenceOperator {
    pub input: VectorFieldID,
    pub output: ScalarFieldID,
    pub parameter_layout: ParameterLayout,
}

impl KernelOperation for DivergenceOperator {
    fn name(&self) -> &'static str {
        "Divergence operator"
    }

    fn validate(&self, state: &SimulationState) -> anyhow::Result<()> {
        let input_meta = state.get_vector_field_metadata(self.input)?;
        let output_meta = state.get_scalar_field_metadata(self.output)?;

        let target_unit = SIUnitDimensions::div(&input_meta.si_unit);
        if SIUnitDimensions::div(&input_meta.si_unit) != output_meta.si_unit {
            return Err(Error::msg(format!("Validation error: incompatible SI units: {} does not match expected {}", output_meta.si_unit, target_unit)));
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
            ],
        };

        parameter_requirements.validate(state.parameter_ref(), &self.parameter_layout)?;

        Ok(())
    }
    
    fn create_kernel(&self, device: &Arc<Device>, state: &SimulationState) -> anyhow::Result<Box<dyn ComputeKernel>> {
        let output_field = state.get_scalar_field(self.output)?;

        match state.get_vector_field(self.input)? {
            Scattered(input_fields) => divergence_op_scattered_input(device, &self.parameter_layout, input_fields, output_field),
            Collocated(_input_field) => todo!(),
        }
    }
    
    fn get_param_layout(&self) -> &ParameterLayout {
        &self.parameter_layout
    }    
}

fn divergence_op_scattered_input(device: &Arc<Device>, parameter_layout: &ParameterLayout, input_fields: &[Field; 2], output: &Field) -> anyhow::Result<Box<dyn ComputeKernel>> {
    let u32_params = parameter_layout.create_u32_buffers(device);
    let f32_params = parameter_layout.create_f32_buffer(device);

    let mut layout_buffer_entries = Vec::new();
    init_helper::add_layout_buffer(&mut layout_buffer_entries, 0, true);
    init_helper::add_layout_buffer(&mut layout_buffer_entries, 1, true);
    init_helper::add_layout_buffer(&mut layout_buffer_entries, 2, false);
    init_helper::add_layout_buffer(&mut layout_buffer_entries, 3, true);
    init_helper::add_layout_buffer(&mut layout_buffer_entries, 4, true);

    Ok(Box::new(LeafKernel::new("Divergence computation")
        .build_pipeline(
            device,
            "shaders/divergence_computation.wgsl",
            "compute_divergence",
            &layout_buffer_entries,
            &vec![
                (0, input_fields[0].get_buffer_ref()),
                (1, input_fields[1].get_buffer_ref()),
                (2, output.get_buffer_ref()),
                (3, &u32_params),
                (4, &f32_params)
            ],
            (input_fields[0].get_buffer_ref().size() + input_fields[1].get_buffer_ref().size()) / std::mem::size_of::<f32>() as u64
        )?
        .save_helper_buffer(u32_params)
        .save_helper_buffer(f32_params)
    ))
}