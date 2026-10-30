use anyhow::Error;

use crate::{ParameterLayout, SIUnitDimensions, state::parameter_registry::ParameterRegistry};



pub struct ParameterRequirementLayout {
    pub u32_requirements: Vec<ParameterRequirement>,
    pub f32_requirements: Vec<ParameterRequirement>,
}

impl ParameterRequirementLayout {
    pub fn validate(&self, registry: &ParameterRegistry, layout: &ParameterLayout) -> anyhow::Result<()> {
        for (requirement, lookup_result) in self.u32_requirements.iter()
            .zip(layout.get_u32_params().iter()
                .map(|id| registry.get_param_u32_units(id.clone()))
            )
        {
            let found_unit = lookup_result?;
            if requirement.units != found_unit {
                return Err(Error::msg(format!("Parameter layout requirement {} not met: {} != {}", requirement.name, requirement.units, found_unit)));
            }
        }

        Ok(())
    }
}

pub struct ParameterRequirement {
    pub name: &'static str,
    pub units: SIUnitDimensions,
}